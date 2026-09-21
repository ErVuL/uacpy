"""Tests for ``uacpy.acoustic_signal.arrays`` on a synthetic line array.

Steering vectors and the Bartlett / MVDR / MUSIC beamformers; ``beamform``
and its shading; ``beamform_field`` over a whole field grid, narrowband and
broadband, and the reception it synthesises; and the three scalars that
bound a scan — ``plane_wave_array_gain``, ``matched_replica_gain`` and
``independent_beams``.
"""

import matplotlib.pyplot as plt
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
from uacpy.acoustic_signal.arrays import (_shaded_steering, beamform,
                                          beamform_field,
                                          independent_beams,
                                          matched_replica_gain,
                                          plane_wave_array_gain)

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


def _pattern_dB(n, d, steer_deg=0.0, weights=None,
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
        ang, pdb = _pattern_dB(self.N, self.D)
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
        ang, pdb = _pattern_dB(self.N, self.D)
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

    def test_dip_at_rayleigh_separation_is_0p9_dB(self):
        ang, p = self._scan(1.0)
        peaks = _local_maxima(p)
        assert len(peaks) == 2                       # still two maxima
        mid = p[np.argmin(np.abs(ang))]
        dip_dB = 10.0 * np.log10(p[peaks].max() / mid)
        # Measured 0.915 dB on this pattern; the doc rounds to 0.9.
        assert dip_dB == pytest.approx(0.92, abs=0.05)

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
        ang, pdb = _pattern_dB(self.N, d, steer_deg=steer)
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
        ang, pdb = _pattern_dB(self.N, d, steer_deg=steer)
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

    @pytest.mark.parametrize('d_frac,gain_dB', [
        (0.5, 12.04),      # = 10·log10(16): isotropic noise spatially white
        (0.25, 9.12),
        (0.125, 6.23),
    ])
    def test_broadside_gain_against_isotropic_noise(self, d_frac, gain_dB):
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
        assert 10.0 * np.log10(ag) == pytest.approx(gain_dB, abs=0.01)

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
        return _pattern_dB(self.N, self.D, weights=w,
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

    @pytest.mark.parametrize('window,loss_dB', [('hann', -1.90),
                                                (('chebwin', 50), -1.54),
                                                ('boxcar', 0.0)])
    def test_white_noise_gain_loss_formula(self, window, loss_dB):
        # ΔG = 10·log10((Σw)² / (N·Σw²)) — the arrays.md closed form.
        w = shading_taper(self.N, window)
        dg = 10.0 * np.log10(w.sum() ** 2 / (self.N * np.sum(w ** 2)))
        assert dg == pytest.approx(loss_dB, abs=0.01)


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


class TestBeamformFieldGoesBroadband:
    """Steering a band means steering every bin, not the centre once.

    A beam delay is a frequency-dependent phase, so one steering vector at
    the band centre mis-steers both edges. And a time-domain reception needs
    the COMPLEX beam output, which a power-only result throws away.
    """

    @staticmethod
    def _band(n_el=16, n_f=33):
        rng = np.random.default_rng(7)
        H = (rng.standard_normal((n_el, n_f))
             + 1j * rng.standard_normal((n_el, n_f)))
        return H, np.linspace(150.0, 250.0, n_f)

    def test_each_bin_is_steered_at_its_own_frequency(self):
        pos, ang = _array(), np.linspace(-30.0, 30.0, 25)
        H, freqs = self._band()
        out = beamform_field(H, pos, ang, freqs, c=C)
        expected = np.empty((ang.size, freqs.size), dtype=complex)
        for i, f in enumerate(freqs):
            e = steering_vectors(pos, ang, f, C)
            expected[:, i] = e.conj() @ H[:, i]
        np.testing.assert_allclose(out.response, expected, atol=1e-12)

    def test_it_differs_from_steering_once_at_the_band_centre(self):
        pos, ang = _array(), np.linspace(-30.0, 30.0, 25)
        H, freqs = self._band()
        broadband = beamform_field(H, pos, ang, freqs, c=C)
        centre = steering_vectors(pos, ang, float(np.mean(freqs)), C).conj() @ H
        assert not np.allclose(broadband.response, centre, atol=1e-6)

    def test_the_response_is_complex_and_power_is_its_modulus_squared(self):
        pos, ang = _array(), np.linspace(-30.0, 30.0, 9)
        H, freqs = self._band()
        out = beamform_field(H, pos, ang, freqs, c=C)
        assert np.iscomplexobj(out.response)
        np.testing.assert_allclose(out.power, np.abs(out.response) ** 2)

    def test_a_frequency_axis_that_does_not_match_is_refused(self):
        pos, ang = _array(), np.linspace(-30.0, 30.0, 9)
        H, freqs = self._band()
        with pytest.raises(ConfigurationError, match='frequency'):
            beamform_field(H[:, :-1], pos, ang, freqs, c=C)

    def test_a_single_frequency_keeps_the_narrowband_shape(self):
        """One frequency must not grow a length-1 frequency axis."""
        pos, ang = _array(), np.linspace(-30.0, 30.0, 9)
        H, _ = self._band()
        out = beamform_field(H[:, 0], pos, ang, FREQ, c=C)
        assert out.response.shape == (ang.size,)
        assert out.frequencies is None


class TestBeamformedFieldDrawsItsBeam:
    """The receive dual of ``plot_beam_pattern``.

    That plotter is a TRANSMIT one: it labels the axis 'Launch angle' and
    warns that a table not spanning +/-90 deg leaves a launch fan
    undefined. Feeding a receive beam to it mislabels the picture and
    raises an irrelevant warning, so the beam power gets its own.
    """

    @staticmethod
    def _beams(n_r=0):
        """One planted arrival, or a grid whose columns differ.

        A grid built by repeating one column cannot tell at=3 from at=0, so
        each column here is a plane wave from its OWN direction and the
        peak names the column.
        """
        pos = _array()
        ang = np.linspace(-45.0, 45.0, 181)
        k = 2.0 * np.pi * FREQ / C
        if n_r == 0:
            p = np.exp(-1j * k * pos * np.sin(np.deg2rad(12.0)))
        else:
            bearings = np.linspace(-30.0, 30.0, n_r)
            p = np.exp(-1j * k * np.outer(pos, np.sin(np.deg2rad(bearings))))
        return beamform_field(p, pos, ang, FREQ, c=C,
                              weights=shading_taper(16, 'hann'))

    def test_it_draws_one_curve_against_the_look_angle(self):
        fig, ax = self._beams().plot()
        assert len(ax.get_lines()) == 1
        x, y = ax.get_lines()[0].get_data()
        np.testing.assert_allclose(x, self._beams().angles)
        assert y.max() == pytest.approx(0.0)          # dB re max
        assert 'angle' in ax.get_xlabel().lower()
        assert 'launch' not in ax.get_xlabel().lower()
        plt.close(fig)

    def test_the_peak_sits_at_the_planted_direction(self):
        fig, ax = self._beams().plot()
        x, y = ax.get_lines()[0].get_data()
        assert x[np.argmax(y)] == pytest.approx(12.0, abs=0.6)
        plt.close(fig)

    def test_normalise_false_keeps_the_absolute_level(self):
        beams = self._beams()
        fig, ax = beams.plot(normalise=False)
        _, y = ax.get_lines()[0].get_data()
        np.testing.assert_allclose(y, 10.0 * np.log10(beams.power))
        plt.close(fig)

    def test_a_grid_needs_a_point_selected(self):
        with pytest.raises(ConfigurationError, match='at='):
            self._beams(n_r=7).plot()

    def test_at_selects_one_point_of_the_grid(self):
        """Each column is a plane wave from its own bearing, so the peak
        of the drawn curve says which column was taken."""
        bearings = np.linspace(-30.0, 30.0, 7)
        beams = self._beams(n_r=7)
        for idx in (0, 3, 6):
            fig, ax = beams.plot(at=idx)
            assert len(ax.get_lines()) == 1
            x, y = ax.get_lines()[0].get_data()
            assert x[np.argmax(y)] == pytest.approx(bearings[idx], abs=0.6)
            plt.close(fig)

    def test_it_overlays_on_a_shared_axes(self):
        fig, ax = plt.subplots()
        self._beams().plot(ax=ax, label='one')
        self._beams().plot(ax=ax, label='two')
        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_a_broadband_beam_needs_its_bin_chosen(self):
        pos = _array()
        freqs = np.linspace(180.0, 220.0, 5)
        H = np.ones((16, freqs.size), dtype=complex)
        beams = beamform_field(H, pos, np.linspace(-45, 45, 91), freqs, c=C)
        with pytest.raises(ConfigurationError, match='at='):
            beams.plot()


class TestBeamformedFieldSynthesisesAReception:
    """The point of keeping the complex response: a beam's time series."""

    @staticmethod
    def _beam():
        pos = _array()
        freqs = np.linspace(150.0, 250.0, 65)
        rng = np.random.default_rng(3)
        H = (rng.standard_normal((16, freqs.size))
             + 1j * rng.standard_normal((16, freqs.size))) * 1e-3
        return beamform_field(H, pos, np.array([-10.0, 0.0, 10.0]), freqs,
                              c=C, weights=shading_taper(16, 'hann'))

    def test_it_returns_a_time_field(self):
        tr = self._beam().to_time_trace(0.0, range_m=5000.0)
        assert list(tr.coords) == ['time']
        assert np.asarray(tr.data).size > 1

    def test_it_matches_the_hand_built_field_route(self):
        from uacpy.core.results import Field
        beams = self._beam()
        i = int(np.argmin(np.abs(beams.angles - 10.0)))
        mine = np.asarray(beams.to_time_trace(10.0, range_m=5000.0).data)
        hand = Field(data=beams.response[i][None, None, :],
                     coords={'depth': np.array([0.0]),
                             'range': np.array([5000.0]),
                             'frequency': beams.frequencies},
                     model='x').to_time_trace(depth=0.0, range=5000.0)
        np.testing.assert_allclose(mine, np.asarray(hand.data), atol=1e-18)

    def test_a_source_spectrum_shapes_the_reception(self):
        beams = self._beam()
        plain = np.asarray(beams.to_time_trace(0.0, range_m=5000.0).data)
        S = np.exp(-((beams.frequencies - 200.0) / 20.0) ** 2)
        shaped = np.asarray(beams.to_time_trace(
            0.0, range_m=5000.0, source_spectrum=S).data)
        assert not np.allclose(plain, shaped)

    def test_the_range_must_be_given(self):
        """There is no separation at which a default would be right."""
        beams = self._beam()
        with pytest.raises(TypeError, match='range_m'):
            beams.to_time_trace(0.0)

    def test_a_narrowband_beam_cannot_make_a_trace(self):
        pos = _array()
        p = np.ones((16, 4), dtype=complex)
        narrow = beamform_field(p, pos, np.array([0.0]), FREQ, c=C)
        with pytest.raises(ConfigurationError, match='frequency'):
            narrow.to_time_trace(0.0, range_m=5000.0)

    def test_extra_grid_axes_are_refused_with_advice(self):
        pos = _array()
        freqs = np.linspace(150.0, 250.0, 17)
        H = np.ones((16, 5, freqs.size), dtype=complex)    # 5 ranges
        beams = beamform_field(H, pos, np.array([0.0]), freqs, c=C)
        with pytest.raises(ConfigurationError, match='one point'):
            beams.to_time_trace(0.0, range_m=5000.0)


class TestWeightsMayBeComplex:
    """A shading is not always real.

    A fixed phase taper, a null steered off the scan grid, an adaptive
    weight vector — all are complex, and casting them to float silently
    discards the imaginary part and returns a plausible wrong number.
    """

    @staticmethod
    def _complex_taper(n=16):
        return shading_taper(n, 'hann') * np.exp(1j * np.linspace(0.0, 2.0, n))

    def test_a_complex_taper_is_not_truncated_to_its_real_part(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 361)
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(10.0)))[:, None]
        w = self._complex_taper()
        full = beamform_field(p, pos, ang, FREQ, c=C, weights=w)
        real_only = beamform_field(p, pos, ang, FREQ, c=C, weights=w.real)
        assert not np.allclose(full.power, real_only.power), \
            'the imaginary part was discarded'

    def test_a_complex_taper_keeps_the_noise_gain_at_one(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 31)
        e = _shaded_steering(pos, ang, FREQ, C, self._complex_taper(), 'test')
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-12)

    def test_plane_wave_array_gain_takes_a_complex_taper(self):
        w = self._complex_taper()
        got = plane_wave_array_gain(w)
        expected = 10.0 * np.log10(np.abs(np.sum(w)) ** 2
                                   / np.sum(np.abs(w) ** 2))
        assert got == pytest.approx(expected)
        assert got != pytest.approx(plane_wave_array_gain(w.real))

    def test_a_phase_ramp_lowers_the_BROADSIDE_gain(self):
        """|sum w| shrinks as a ramp spreads the phases: a real effect."""
        pos = _array()
        n = pos.size
        w = np.ones(n) * np.exp(1j * np.linspace(0.0, 3.0, n))
        assert plane_wave_array_gain(w) < plane_wave_array_gain(np.ones(n))

    def test_the_two_gains_answer_different_questions(self):
        """Broadside gain is not the gain on the wave the taper is matched to.

        A complex taper steers, so its matched wave is not at broadside.
        ``plane_wave_array_gain`` reports broadside; the scan finds the
        steered direction. On an 8-radian ramp they differ by ~10 dB, and
        making them agree would erase the superdirective case.
        """
        pos = _array()
        ang = np.linspace(-90.0, 90.0, 2881)      # fine enough to find it
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(0.0))[:, None]     # broadside wave
        w = shading_taper(16, 'hann') * np.exp(-1j * np.linspace(0.0, 8.0, 16))
        broadside = plane_wave_array_gain(w)
        scan = beamform_field(p, pos, ang, FREQ, c=C, weights=w)
        assert broadside < 1.0                    # the ramp nulls broadside
        assert scan.array_gain()[0] > 9.0         # the scan finds the lobe
        assert scan.array_gain()[0] - broadside > 9.0
        assert abs(scan.best_angle[0]) > 5.0      # and it is not at 0 deg

    def test_it_reproduces_butler_and_shermans_superdirective_pair(self):
        """B&S 8.4.1, five hydrophones, incoherent noise: +7 and -7 dB.

        Their superdirective shading is ``a_i = 1`` for the centre and the
        two ends, ``-1`` for the other two — alternating, and with FIVE
        elements it sums to 1, not 0. "The array gain goes from 7 dB for
        uniform shading to -7 dB for the simple superdirective shading."
        A magnitudes-only formula would report +7 dB for both.
        """
        # 10log10(5) = 6.9897, which B&S quote as 7 dB; assert the exact
        # value so the test pins the formula, not their rounding.
        exact = 10.0 * np.log10(5.0)
        assert plane_wave_array_gain(np.ones(5)) == pytest.approx(exact)
        superdirective = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        assert np.sum(superdirective) == 1.0        # not zero: five elements
        assert plane_wave_array_gain(superdirective) == pytest.approx(-exact)

    def test_weights_that_sum_to_zero_null_the_steered_direction(self):
        """An EVEN alternating array sums to zero, which is a true null.

        Not Butler & Sherman's case: theirs has five elements and sums to
        1. The parity matters — the answer is -inf here and -7 dB there.
        """
        w = np.array([1.0, -1.0] * 8)
        assert np.sum(w) == 0.0
        assert plane_wave_array_gain(w) == -np.inf

    def test_all_zero_weights_are_refused_not_silently_nan(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 9)
        p = np.ones((16, 3), dtype=complex)
        with pytest.raises(ConfigurationError, match='no power'):
            beamform_field(p, pos, ang, FREQ, c=C, weights=np.zeros(16))

    def test_non_numeric_weights_are_refused(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 9)
        p = np.ones((16, 3), dtype=complex)
        with pytest.raises(ConfigurationError, match='weights'):
            beamform_field(p, pos, ang, FREQ, c=C,
                           weights=np.array(['a'] * 16))


class TestShadedSteeringIsRenormalised:
    """The unit row norm must bite for a taper that is not already normalised.

    ``shading_taper`` returns ``w / sqrt(mean(w**2))``, so ``||w||**2 == N``
    for every window it makes, and multiplying a ``1/sqrt(N)`` steering row
    by it lands on unit norm by accident. A test fed only those tapers
    cannot see the re-normalisation at all — deleting the line leaves the
    whole file green. ``np.hanning`` has ``||w||**2 = 5.625`` at N=16, which
    is what makes these tests able to fail.
    """

    @staticmethod
    def _raw_taper(n=16):
        w = np.hanning(n)
        assert not np.isclose(np.sum(w ** 2), n), 'fixture must not be pre-normalised'
        return w

    def test_a_raw_taper_leaves_the_noise_gain_at_one(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 31)
        w = self._raw_taper()
        e = _shaded_steering(pos, ang, FREQ, C, w, 'test')
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-12)

    def test_a_raw_taper_gives_the_same_gain_as_the_normalised_one(self):
        """AG is scale-invariant, so the two spellings must agree exactly."""
        pos, ang = _array(), np.linspace(-45.0, 45.0, 361)
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(0.0)))[:, None]
        raw = beamform_field(p, pos, ang, FREQ, c=C, weights=self._raw_taper())
        tapered = beamform_field(p, pos, ang, FREQ, c=C,
                                 weights=shading_taper(16, 'hann'))
        np.testing.assert_allclose(raw.array_gain(), tapered.array_gain(),
                                   atol=1e-9)

    def test_beamform_peak_is_the_taper_gain_for_a_raw_taper(self):
        """Without the re-normalisation this peak is wrong by 10log10(N/||w||^2)."""
        pos = _array()
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(0.0)))[:, None]
        w = self._raw_taper()
        res = beamform_fn(p, pos, FREQ, SL=0.0, NL=0.0, weights=w)
        assert res.peak_snr == pytest.approx(plane_wave_array_gain(w), abs=1e-9)


class TestBeamformFieldCoversAWholeGrid:
    """A coverage map needs the beamformer run over every point of a field.

    ``beamform`` takes one ``(n_phones, n_ranges)`` slice and returns SNR in
    dB, so a depth-range plane, the beam power itself, and the look angle
    that won had to be assembled by hand at every call site.
    """

    @staticmethod
    def _plane(n_el=16, n_z=7, n_r=11, seed=3):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((n_el, n_z, n_r))
                + 1j * rng.standard_normal((n_el, n_z, n_r)))

    def test_it_matches_the_hand_rolled_contraction(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 61)
        p = self._plane()
        taper = shading_taper(16, 'hann')
        out = beamform_field(p, pos, ang, FREQ, c=C, weights=taper)
        W = steering_vectors(pos, ang, FREQ, C) * taper[None, :]
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        expected = np.abs(np.einsum('ae,ezr->azr', W.conj(), p)) ** 2
        np.testing.assert_allclose(out.power, expected, rtol=1e-12, atol=1e-12)
        assert out.power.shape == (ang.size, 7, 11)

    def test_a_one_dimensional_grid_keeps_its_shape(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 61)
        p = self._plane()[:, 0, :]
        out = beamform_field(p, pos, ang, FREQ, c=C)
        assert out.power.shape == (ang.size, p.shape[1])
        assert out.element_power.shape == (p.shape[1],)

    def test_best_angle_recovers_a_planted_plane_wave(self):
        pos = _array()
        ang = np.linspace(-45.0, 45.0, 181)     # 0.5 deg grid
        k = 2.0 * np.pi * FREQ / C
        for truth in (-20.0, 0.0, 12.5):
            p = np.exp(-1j * k * pos * np.sin(np.deg2rad(truth)))[:, None]
            out = beamform_field(p, pos, ang, FREQ, c=C)
            assert out.best_angle[0] == pytest.approx(truth, abs=0.5)

    def test_array_gain_of_a_matched_plane_wave_is_the_weight_vector_gain(self):
        pos = _array()
        ang = np.linspace(-45.0, 45.0, 361)
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(0.0)))[:, None]
        for window in ('boxcar', 'hann'):
            w = shading_taper(16, window)
            out = beamform_field(p, pos, ang, FREQ, c=C, weights=w)
            assert out.array_gain()[0] == pytest.approx(
                plane_wave_array_gain(w), abs=1e-9)

    def test_the_element_count_is_checked(self):
        with pytest.raises(ConfigurationError, match='element'):
            beamform_field(self._plane(n_el=15), _array(),
                           np.linspace(-45.0, 45.0, 9), FREQ, c=C)


class TestPlaneWaveArrayGainIsTheWeightVectorRatio:
    """``AG = |sum w|^2 / ||w||^2`` — not ``-10log10(sum |w|^4)``.

    The two agree for an unshaded array and differ by about 1.1 dB for a
    Hann one, which is the shape of error that reads as plausible.
    """

    def test_an_unshaded_array_gives_10log10_n(self):
        for n in (4, 16, 24):
            assert plane_wave_array_gain(np.ones(n)) == pytest.approx(
                10.0 * np.log10(n))

    def test_a_hann_taper_costs_about_two_dB(self):
        n = 24
        loss = (10.0 * np.log10(n)
                - plane_wave_array_gain(shading_taper(n, 'hann')))
        assert loss == pytest.approx(1.95, abs=0.05)

    def test_it_is_scale_invariant(self):
        w = shading_taper(24, 'hann')
        assert plane_wave_array_gain(7.3 * w) == pytest.approx(
            plane_wave_array_gain(w))

    def test_it_differs_from_the_sum_of_fourth_powers(self):
        w = shading_taper(24, 'hann')
        wn = w / np.linalg.norm(w)
        wrong = -10.0 * np.log10(np.sum(np.abs(wn) ** 4))
        assert abs(plane_wave_array_gain(w) - wrong) > 1.0


class TestMatchedReplicaGainIsTheWhiteNoiseCeiling:
    """A replica matched to the field realises ``10log10(N)`` exactly.

    Whatever shape the field has: that is what makes it the bound a
    conventional beamformer is measured against.
    """

    def test_it_equals_10log10_n_for_any_field(self):
        rng = np.random.default_rng(11)
        for shape in ((24, 200), (16, 7, 11), (8,)):
            p = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            got = matched_replica_gain(p)
            np.testing.assert_allclose(got, 10.0 * np.log10(shape[0]),
                                       rtol=0, atol=1e-9)

    def test_it_bounds_what_a_plane_wave_scan_realises(self):
        pos, ang = _array(), np.linspace(-45.0, 45.0, 361)
        rng = np.random.default_rng(5)
        p = (rng.standard_normal((16, 40))
             + 1j * rng.standard_normal((16, 40)))      # a multi-mode arrival
        scan = beamform_field(p, pos, ang, FREQ, c=C).array_gain()
        assert np.all(scan <= matched_replica_gain(p) + 1e-9)


class TestIndependentBeamsCountsOrthogonalLooks:
    """Beams spaced ``lambda / (N*d)`` in ``sin(theta)`` are orthogonal.

    The aperture form ``lambda / ((N-1)*d)`` is the tempting one and leaves
    the neighbours correlated — |corr| = 1/N, so 0.0625 on this file's
    16-element fixture, against 1e-16 at the orthogonal spacing. Being the
    WIDER spacing it also yields fewer cells: 10.61 against 11.31 over
    +/-45 deg here (16.26 against 16.97 for the 24-element array in
    example 42). A threshold set from it is set for fewer looks than the
    detector takes, and is optimistic.
    """

    def test_the_spacing_it_implies_is_orthogonal(self):
        pos = _array()
        ang = np.linspace(-90.0, 90.0, 361)
        n = independent_beams(pos, ang, FREQ, c=C)
        span = np.ptp(np.sin(np.deg2rad(ang)))
        k = 2.0 * np.pi * FREQ / C
        a0 = np.exp(-1j * k * pos * 0.0)
        a1 = np.exp(-1j * k * pos * (span / n))
        assert abs(np.vdot(a0, a1)) / pos.size < 1e-12

    def test_a_full_visible_sector_holds_about_n_beams(self):
        pos = _array()                       # 16 elements at lambda/2
        ang = np.linspace(-90.0, 90.0, 721)
        assert independent_beams(pos, ang, FREQ, c=C) == pytest.approx(
            16.0, rel=1e-9)

    def test_a_boxcar_taper_reproduces_the_unshaded_count_exactly(self):
        """The shaded path must reduce to the unshaded one, not merely near it."""
        pos, ang = _array(), np.linspace(-90.0, 90.0, 721)
        plain = independent_beams(pos, ang, FREQ, c=C)
        boxcar = independent_beams(pos, ang, FREQ, c=C,
                                   weights=shading_taper(16, 'boxcar'))
        assert boxcar == pytest.approx(plain, rel=1e-3)

    def test_a_taper_widens_the_cell_and_so_lowers_the_count(self):
        """Shading spends aperture on sidelobes, so cells get wider.

        The beams' NOISE correlation is the array factor of |w|^2, so the
        width that matters is that of hann^2 — first null at three DFT bins
        against the rectangular window's one, not the two bins of the beam
        PATTERN. The looks a scan really has are fewer than the resolution
        argument suggests. Blackman is wider still.
        """
        pos, ang = _array(), np.linspace(-90.0, 90.0, 721)
        plain = independent_beams(pos, ang, FREQ, c=C)
        hann = independent_beams(pos, ang, FREQ, c=C,
                                 weights=shading_taper(16, 'hann'))
        black = independent_beams(pos, ang, FREQ, c=C,
                                  weights=shading_taper(16, 'blackman'))
        assert hann == pytest.approx(plain / 3.20, rel=0.02)
        assert black < hann < plain

    def test_the_hann_widening_tends_to_three_bins_as_the_array_grows(self):
        """3.20x at 16 elements is a finite-N effect; hann^2 gives 3."""
        ang = np.linspace(-90.0, 90.0, 721)
        d = 0.5 * C / FREQ
        w16 = independent_beams(d * np.arange(16), ang, FREQ, c=C) / \
            independent_beams(d * np.arange(16), ang, FREQ, c=C,
                              weights=shading_taper(16, 'hann'))
        w64 = independent_beams(d * np.arange(64), ang, FREQ, c=C) / \
            independent_beams(d * np.arange(64), ang, FREQ, c=C,
                              weights=shading_taper(64, 'hann'))
        assert 3.0 < w64 < w16 < 3.3

    def test_a_pure_phase_ramp_does_not_change_the_count(self):
        """Steering a taper moves the beam; it does not widen it."""
        pos, ang = _array(), np.linspace(-90.0, 90.0, 721)
        t = shading_taper(16, 'hann')
        straight = independent_beams(pos, ang, FREQ, c=C, weights=t)
        steered = independent_beams(
            pos, ang, FREQ, c=C,
            weights=t * np.exp(-1j * np.linspace(0.0, 3.0, 16)))
        assert steered == pytest.approx(straight, rel=1e-6)

    def test_a_narrower_scan_holds_proportionally_fewer(self):
        pos = _array()
        wide = independent_beams(pos, np.linspace(-90.0, 90.0, 721), FREQ, c=C)
        half = independent_beams(pos, np.linspace(-30.0, 30.0, 721), FREQ, c=C)
        assert half == pytest.approx(wide * np.sin(np.deg2rad(30.0)), rel=1e-9)


class TestBeamformTakesAShadingTaper:
    """A shaded beamformer trades main-lobe width and gain for sidelobes.

    Without a ``weights`` argument the only way to shade is to rebuild the
    weight matrix by hand, which puts the unit-norm convention — the one
    that keeps the noise gain at 1 — on every caller.
    """

    def test_a_uniform_taper_changes_nothing(self):
        pos = _array()
        rng = np.random.default_rng(5)
        p = (rng.standard_normal(16) + 1j * rng.standard_normal(16))[:, None]
        plain = beamform_fn(p, pos, FREQ, SL=0.0)
        boxcar = beamform_fn(p, pos, FREQ, SL=0.0, weights=np.ones(16))
        np.testing.assert_allclose(boxcar.snr, plain.snr, atol=1e-12)

    def test_a_hann_taper_lowers_the_peak_by_its_own_loss(self):
        pos = _array()
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(0.0)))[:, None]
        w = shading_taper(16, 'hann')
        shaded = beamform_fn(p, pos, FREQ, SL=0.0, NL=0.0, weights=w)
        # A matched plane wave gives |sum w|^2 / ||w||^2 for unit-norm w,
        # which is the array gain the taper leaves.
        wn = w / np.linalg.norm(w)
        expected = 10.0 * np.log10(np.abs(np.sum(wn)) ** 2
                                   / np.sum(np.abs(wn) ** 2))
        assert shaded.peak_snr == pytest.approx(expected, abs=1e-9)

    def test_the_taper_buys_lower_sidelobes(self):
        pos = _array()
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(0.0))[:, None]
        ang = np.linspace(-90.0, 90.0, 721)
        plain = beamform_fn(p, pos, FREQ, angles=ang, SL=0.0, NL=0.0)
        shaded = beamform_fn(p, pos, FREQ, angles=ang, SL=0.0, NL=0.0,
                             weights=shading_taper(16, 'hann'))
        far = np.abs(ang) > 30.0          # well outside either main lobe
        assert shaded.snr[far, 0].max() < plain.snr[far, 0].max() - 5.0

    def test_a_taper_that_does_not_fit_the_array_is_refused(self):
        p = np.ones((16, 1), dtype=complex)
        with pytest.raises(ConfigurationError, match='weights'):
            beamform_fn(p, _array(), FREQ, weights=np.ones(15))


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
