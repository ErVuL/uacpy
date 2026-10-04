"""``SoundSpeedProfile`` and the sea-surface generator beside it in
``uacpy.core.ssp``.

Nearest versus interpolated reads, depth-only slices of a range-dependent
profile, ``extend_to``, the formula a copy keeps, and
``generate_sea_surface``'s synthesis and its sampling warning.
"""

import inspect
import numpy as np
import pytest
import uacpy
import warnings
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.ssp import SoundSpeedProfile


class TestGenerateSeaSurfaceSynthesis:
    """The realisation is the inverse DFT of the Pierson-Moskowitz
    random-phase spectrum, so it must equal the direct cosine sum it is
    defined by and stay seed-reproducible."""

    def test_matches_the_direct_cosine_sum(self):
        from uacpy.core.altimetry import generate_sea_surface
        from uacpy.core.units import ms_to_knots
        n, max_range, wind, seed = 257, 4000.0, 10.0, 12345     # wind in m/s
        out = generate_sea_surface(max_range, ms_to_knots(wind), n,
                                   rng=np.random.default_rng(seed))
        ranges, surface = out[:, 0], out[:, 1]
        dx = ranges[1] - ranges[0]
        dk = 1.0 / (n * dx)
        k = np.arange(1, n // 2 + 1) * dk
        from uacpy.core.constants import STANDARD_GRAVITY_M_S2
        g = STANDARD_GRAVITY_M_S2
        omega = np.sqrt(g * 2 * np.pi * k)
        omega_p = g / wind
        S_omega = (8.1e-3 * g ** 2 / omega ** 5) * np.exp(
            -0.74 * (omega_p / omega) ** 4)
        S_k = S_omega * (np.pi * g / omega)
        amplitude = np.sqrt(2 * S_k * dk)
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(k))
        direct = (amplitude[None, :]
                  * np.cos(2 * np.pi * ranges[:, None] * k[None, :]
                           + phase[None, :])).sum(axis=1)
        assert float(np.max(np.abs(surface - direct))) < 1e-9

    def test_from_sea_state_carries_the_generated_samples(self):
        """``Altimetry.from_sea_state`` is the carrier form of
        ``generate_sea_surface``: every argument reaches the generator, and
        the carrier holds its samples bit for bit."""
        from uacpy.core.altimetry import Altimetry, generate_sea_surface
        pairs = generate_sea_surface(3000.0, 12.0, 300,
                                     rng=np.random.default_rng(5))
        alt = Altimetry.from_sea_state(3000.0, 12.0, 300,
                                       rng=np.random.default_rng(5))
        assert isinstance(alt, Altimetry)
        assert np.array_equal(alt.ranges, pairs[:, 0])
        assert np.array_equal(alt.heights, pairs[:, 1])

    def test_seeded_runs_are_reproducible(self):
        from uacpy.core.altimetry import generate_sea_surface
        a = generate_sea_surface(2000.0, 10.0, 128,
                                 rng=np.random.default_rng(7))
        b = generate_sea_surface(2000.0, 10.0, 128,
                                 rng=np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_the_draw_comes_from_the_generator_passed(self):
        """``rng=`` is a Generator, like every other drawing function; an
        int is refused rather than read as a seed, and the positional
        fourth argument the seed used to occupy is gone."""
        from uacpy.core.altimetry import generate_sea_surface
        rng = np.random.default_rng(7)
        first = generate_sea_surface(2000.0, 10.0, 128, rng=rng)
        second = generate_sea_surface(2000.0, 10.0, 128, rng=rng)
        assert not np.array_equal(first, second)
        with pytest.raises(ConfigurationError, match='default_rng'):
            generate_sea_surface(2000.0, 10.0, 128, rng=7)
        with pytest.raises(TypeError,
                           match='positional arguments but 4 were given'):
            generate_sea_surface(2000.0, 10.0, 128, 7)


class TestSoundSpeedProfileNearestVsInterp:
    """``SoundSpeedProfile.at(...)`` is **nearest** (never fabricates),
    ``.eval(...)`` is **linear**, ``.isel(...)`` is **positional** — the
    grid-library invariant shared with ``Field`` et al."""

    def _ssp(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            sound_speed=np.array([[1500.0], [1490.0], [1480.0]])
        )

    def test_at_picks_nearest_depth(self):
        ssp = self._ssp()
        # depth=51 is closer to 100 than to 0 → returns the 100m sample
        sliced = ssp.at(depth=51.0)
        assert sliced.depths[0] == 100.0
        assert sliced.value == 1490.0

    def test_eval_interpolates_linear(self):
        ssp = self._ssp()
        # depth=50 is halfway between (0, 1500) and (100, 1490) → 1495
        sliced = ssp.eval(depth=50.0)
        assert sliced.depths[0] == 50.0
        assert sliced.value == pytest.approx(1495.0)

    def test_isel_positional_depth(self):
        ssp = self._ssp()
        sliced = ssp.isel(depth=1)
        assert sliced.depths[0] == 100.0 and sliced.value == 1490.0
        with pytest.raises(IndexError, match='depth index 9 out of range'):
            ssp.isel(depth=9)


class TestSoundSpeedProfileDepthOnlySliceOf2D:
    """A depth-only slice of a range-dependent profile is ambiguous.

    Silently returning the r = 0 column is wrong physics on exactly the
    profiles the 2-D carrier exists for, so ``at`` / ``eval`` / ``isel``
    require the range to be pinned (or the range axis collapsed first).
    """

    def _ssp2d(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile.from_2d(
            depths=[0.0, 100.0], ranges=[0.0, 5000.0],
            matrix=[[1500.0, 1400.0], [1490.0, 1390.0]])

    @pytest.mark.parametrize('call', [
        lambda s: s.at(depth=50.0),
        lambda s: s.eval(depth=50.0),
        lambda s: s.isel(depth=0),
    ])
    def test_depth_only_slice_raises(self, call):
        with pytest.raises(ConfigurationError, match='range-dependent'):
            call(self._ssp2d())

    def test_pinning_the_range_works(self):
        s = self._ssp2d()
        assert s.at(depth=0.0, range=5000.0).value == pytest.approx(1400.0)
        assert s.eval(depth=0.0, range=2500.0).value == pytest.approx(1450.0)
        assert s.isel(depth=0, range=1).value == pytest.approx(1400.0)

    def test_collapsing_the_range_axis_works(self):
        assert self._ssp2d().collapse_range('mean').at(depth=0.0).value == \
            pytest.approx(1450.0)

    def test_range_only_and_1d_paths_unaffected(self):
        from uacpy.core.environment import SoundSpeedProfile
        assert self._ssp2d().eval(range=5000.0).sound_speed.shape == (2, 1)
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        assert flat.at(depth=50.0).value == pytest.approx(1500.0)

    @pytest.mark.parametrize('call', [
        lambda s: s.eval(range=0.0, method='bogus'),
        lambda s: s.eval(depth=50.0, method='bogus'),
    ])
    def test_interp_method_validated_on_every_path(self, call):
        """The membership check runs on entry, so the range-independent
        shortcut cannot swallow a bad method name."""
        from uacpy.core.environment import SoundSpeedProfile
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        with pytest.raises(ConfigurationError, match='interpolation method'):
            call(flat)


class TestSoundSpeedProfileExtendTo:
    """``SoundSpeedProfile.extend_to(z_max)`` is the canonical alignment
    hook used by every env writer. Must extend OR truncate so that
    ``ssp.depths[-1] == z_max`` exactly."""

    def _profile(self, depths, speeds):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.asarray(depths, dtype=float),
            sound_speed=np.asarray(speeds, dtype=float).reshape(-1, 1)
        )

    def test_noop_when_depth_max_equals_deepest(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        assert ssp.extend_to(200.0) is ssp

    def test_extend_with_constant_extrapolation(self):
        out = self._profile([0, 100], [1500, 1490]).extend_to(300.0)
        assert out.depths[-1] == 300.0
        assert out.sound_speed[-1, 0] == 1490.0

    def test_truncate_with_linear_interpolation(self):
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.sound_speed[-1, 0] == pytest.approx(1485.0)
        assert (out.depths <= 150.0).all()

    def test_truncate_then_extend_round_trip(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1480])
        out = ssp.extend_to(150.0).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.sound_speed[-1, 0] == pytest.approx(1485.0)

    def test_noop_under_floating_point_drift(self):
        """``extend_to`` is a no-op when the requested depth matches the
        deepest sample to within a small relative tolerance — a 1-ulp
        drift (from e.g. a round trip through I/O) must not rewrite the
        bottom sample."""
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        # Smallest perturbation that survives a few arithmetic ops:
        perturbed = 200.0 + 1e-12
        out = ssp.extend_to(perturbed)
        assert out is ssp

    def test_truncation_snaps_a_sample_inside_the_readers_last_point_window(self):
        """``misc/sspMod.f90:353`` ends a medium's SSP block at the first
        sample within ``AT_LAST_SSP_POINT_EPS_M`` (1.19e-5 m) of the declared
        medium depth. Truncating beside such a sample emits two rows metres
        apart in index and microns apart in depth; the reader takes the first
        as the end of the block and consumes the second as the bottom-option
        record. The sample has to *move* onto the target instead — the same
        rule the near-miss branch applies when no truncation is needed."""
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        out = ssp.extend_to(100.0)
        assert out.depths.tolist() == [0.0, 50.0, 100.0]
        assert out.sound_speed[-1, 0] == pytest.approx(1490.0)
        assert float(np.min(np.diff(out.depths))) > 1.1920929e-05

    def test_the_written_deck_ends_the_block_once(self):
        """The consequence the snap exists for, read off the deck the AT
        writer produces: one last SSP row at the medium depth, not two."""
        import io
        from uacpy.io.oalib_writer import write_ssp_section
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        env = uacpy.Environment(name='snap', bathymetry=100.0, ssp=ssp,
                                bottom=1800.0)
        buf = io.StringIO()
        write_ssp_section(buf, env, 100.0, ssp_topopt='C')
        depths = [float(line.split()[0]) for line in
                  buf.getvalue().splitlines()[1:] if line.strip()]
        assert depths == [0.0, 50.0, 100.0]

    def test_an_ordinary_truncation_interpolates_a_new_sample(self):
        """The discriminating half: when the deepest surviving sample is a
        real distance above the target, the final row is still interpolated
        rather than dragged down onto it."""
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths.tolist() == [0.0, 100.0, 150.0]
        assert out.sound_speed[-1, 0] == pytest.approx(1485.0)

    def test_the_source_profile_is_untouched(self):
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        ssp.extend_to(100.0)
        assert ssp.depths[2] == pytest.approx(99.999995)
        assert ssp.depths.size == 5

    def test_a_range_dependent_profile_snaps_every_column(self):
        from uacpy.core.environment import SoundSpeedProfile
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 50.0, 99.999995, 150.0]),
            sound_speed=np.array([[1500.0, 1502.0], [1495.0, 1497.0],
                           [1490.0, 1492.0], [1485.0, 1487.0]]),
            ranges=np.array([0.0, 5000.0]))
        out = ssp.extend_to(100.0)
        assert out.depths.tolist() == [0.0, 50.0, 100.0]
        assert out.sound_speed[-1].tolist() == [1490.0, 1492.0]
        assert out.ranges.tolist() == [0.0, 5000.0]


class TestSoundSpeedProfileCopiesKeepTheFormula:
    """``formula`` records which seawater equation built ``data``; the deep
    extension in ``uacpy.data.sound_speed`` reads it to continue the column
    under the same equation. Every copy a slicer returns carries it, as
    ``collapse`` and ``extend_to`` do."""

    @staticmethod
    def _profile(range_dependent):
        if range_dependent:
            return SoundSpeedProfile(
                depths=[0.0, 50.0, 100.0], ranges=[0.0, 1000.0],
                sound_speed=[[1500.0, 1510.0], [1495.0, 1505.0], [1490.0, 1500.0]],
                formula='delgrosso')
        return SoundSpeedProfile(
            depths=[0.0, 50.0, 100.0], sound_speed=[1500.0, 1495.0, 1490.0],
            formula='delgrosso')

    @pytest.mark.parametrize('slicer', [
        lambda p: p.at(depth=10.0),
        lambda p: p.eval(depth=10.0),
        lambda p: p.isel(depth=1),
        lambda p: p.at(range=0.0),
        lambda p: p.extend_to(200.0),
    ], ids=['at-depth', 'eval-depth', 'isel-depth', 'at-range', 'extend_to'])
    def test_a_one_dimensional_copy_carries_the_formula(self, slicer):
        assert slicer(self._profile(False)).formula == 'delgrosso'

    @pytest.mark.parametrize('slicer', [
        lambda p: p.at(range=0.0),
        lambda p: p.eval(range=500.0),
        lambda p: p.isel(range=1),
        lambda p: p.at(depth=10.0, range=0.0),
        lambda p: p.eval(depth=10.0, range=500.0),
        lambda p: p.isel(depth=1, range=0),
        lambda p: p.collapse_range(),
        lambda p: p.extend_to(200.0),
    ], ids=['at-range', 'eval-range', 'isel-range', 'at-both', 'eval-both',
            'isel-both', 'collapse', 'extend_to'])
    def test_a_two_dimensional_copy_carries_the_formula(self, slicer):
        assert slicer(self._profile(True)).formula == 'delgrosso'

    def test_a_literal_profile_stays_literal(self):
        ssp = SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1490.0])
        assert ssp.at(depth=10.0).formula is None


class TestSingleNodeRangesTravelsThroughSspSlicing:
    """A single-node ``ranges`` is a coordinate at that range —
    ``env.max_range`` reads it — so every SSP slice of a single-column
    profile carries it, the rule ``Bottom.collapse_range`` states for the
    same case."""

    def _ssp(self):
        return SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                 sound_speed=[[1500.0], [1490.0], [1495.0]],
                                 ranges=[5000.0])

    def test_a_depth_only_slice_keeps_env_max_range(self):
        env = Environment(ssp=self._ssp().at(depth=50.0), bathymetry=100.0)
        assert env.range_max == pytest.approx(5000.0, rel=1e-12)

    @pytest.mark.parametrize('slicer', [
        lambda s: s.at(depth=50.0),
        lambda s: s.at(range=5000.0),
        lambda s: s.eval(depth=25.0),
        lambda s: s.isel(depth=0),
        lambda s: s.isel(range=0),
    ], ids=['at_depth', 'at_range', 'eval_depth', 'isel_depth',
            'isel_range'])
    def test_each_slice_keeps_the_single_node_ranges(self, slicer):
        out = slicer(self._ssp())
        assert out.ranges is not None
        assert float(out.ranges[0]) == pytest.approx(5000.0, rel=1e-12)

    def test_collapsing_a_range_dependent_profile_drops_ranges(self):
        """Pinning the range axis of a multi-column profile collapses it, so
        the result carries no ranges — the ``Bottom.collapse_range('r0')``
        counterpart."""
        rd = SoundSpeedProfile(depths=[0.0, 100.0],
                               sound_speed=[[1500.0, 1510.0], [1490.0, 1505.0]],
                               ranges=[0.0, 5000.0])
        assert rd.at(range=5000.0).ranges is None
        assert rd.isel(range=1).ranges is None


class TestTheSeaSurfaceUndersamplingWarningNamesTheNominalPeak:
    """``generate_sea_surface`` compares the grid's Nyquist wavenumber against
    ``2 * k0``, with ``k0`` built from M&C's nominal ``omega_p = g/W``
    (13.1.11). The code comment says "nominal"; the warning text dropped the
    word and called ``k0`` "the Pierson-Moskowitz peak", which it is not — the
    spectrum ``S ~ omega^-5 exp(-beta (omega_p/omega)^4)`` peaks below it, so
    the guard's margin is wider than the sentence implied. The threshold is a
    calibration and is unchanged; only the sentence is."""

    BETA = 0.74

    def test_the_true_spectral_peak_sits_below_the_nominal_one(self):
        # Stationary point of omega^-5 exp(-beta (w0/w)^4): w^4 = 4*beta/5*w0^4.
        ratio = (4.0 * self.BETA / 5.0) ** 0.25
        assert ratio == pytest.approx(0.8772, abs=5e-5)
        # omega -> k goes as omega^2 under the deep-water dispersion.
        assert ratio ** 2 == pytest.approx(0.7694, abs=5e-5)

    def test_the_two_times_factor_is_really_two_point_six_true_peaks(self):
        # The size of the overstatement, which is why the word matters.
        assert 2.0 / (4.0 * self.BETA / 5.0) ** 0.5 == pytest.approx(2.599,
                                                                     abs=5e-4)

    def test_the_true_peak_is_a_maximum_of_the_spectrum(self):
        # Not merely a stationary point: the discriminating check that the
        # ratio above is the peak and the nominal omega_p is not.
        w0 = 1.0

        def S(w):
            return w ** -5 * np.exp(-self.BETA * (w0 / w) ** 4)

        w_true = (4.0 * self.BETA / 5.0) ** 0.25 * w0
        assert S(w_true) > S(w0)
        assert S(w_true) > S(w_true * 1.01)
        assert S(w_true) > S(w_true * 0.99)

    def test_the_warning_calls_the_threshold_quantity_nominal(self):
        from uacpy.core.altimetry import generate_sea_surface
        # 32 samples over 4 km is a 129 m step against the 18 m that 8
        # samples per 144 m nominal peak wavelength (15 m/s) want.
        with pytest.warns(UserWarning, match='nominal') as rec:
            generate_sea_surface(4000.0, wind_speed_kn=30.0, n_points=32,
                                 rng=np.random.default_rng(1))
        message = str(rec[0].message)
        assert 'omega_p = g/W' in message
        assert 'the *nominal* Pierson-Moskowitz peak' in message

    def test_a_grid_that_resolves_the_peak_stays_silent(self):
        # The far side of the threshold: 8 samples per 64 m peak wavelength
        # over 4 km is 501 points at 10 m/s.
        from uacpy.core.altimetry import generate_sea_surface
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            generate_sea_surface(4000.0, wind_speed_kn=20.0, n_points=501,
                                 rng=np.random.default_rng(1))


class TestAMackenzieProfileIsExtendedUnderMackenzie:
    """``SoundSpeedProfile.from_temperature_salinity`` stamps ``formula='mackenzie'`` and
    ``uacpy.data.extend_ssp_below_data`` reads that stamp, so the deep
    extension continues the column under the equation that built it. Before
    the stamp the column was continued under TEOS-10 (the ``None`` default),
    which sits 0.35 m/s below Mackenzie at 8.8 km from a 5.5 km column."""

    _Z = np.linspace(0.0, 5500.0, 56)
    _T = np.where(_Z < 1000.0, 15.0 - 0.013 * _Z, 2.0)
    _S = np.full(_Z.shape, 35.0)

    def _mackenzie_profile(self):
        # formula= is explicit now: the constructor no longer pins one
        # equation in its name, so the default is the package default
        # (TEOS-10) and Mackenzie is asked for.
        return SoundSpeedProfile.from_temperature_salinity(
            self._Z, self._T, self._S, formula='mackenzie')

    def test_the_constructor_stamps_the_formula_it_was_given(self):
        assert self._mackenzie_profile().formula == 'mackenzie'

    def test_the_extension_matches_mackenzie_at_the_seafloor(self):
        from uacpy.core.acoustics import sound_speed_mackenzie
        from uacpy.data import extend_ssp_below_data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            extended = extend_ssp_below_data(self._mackenzie_profile(), 8800.0)
        # The extension inverts an effective temperature from the deepest
        # sample at S = 35, which is exactly this column's 2 °C, so the
        # continued value is Mackenzie at (2 °C, 35, 8800 m) itself.
        reference = float(sound_speed_mackenzie(temperature=2.0, salinity=35.0,
                                     depth=8800.0))
        assert abs(float(extended.sound_speed[-1, 0]) - reference) < 0.05

    def test_the_same_numbers_without_the_stamp_continue_under_teos10(self):
        from uacpy.core.acoustics import sound_speed_mackenzie
        from uacpy.data import extend_ssp_below_data
        stamped = self._mackenzie_profile()
        literal = SoundSpeedProfile(depths=stamped.depths,
                                    sound_speed=stamped.sound_speed.copy())
        assert literal.formula is None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            extended = extend_ssp_below_data(literal, 8800.0)
        reference = float(sound_speed_mackenzie(temperature=2.0, salinity=35.0,
                                     depth=8800.0))
        assert abs(float(extended.sound_speed[-1, 0]) - reference) > 0.2


class TestTheSeaSurfaceResolutionRuleIsTheWarningsBoundary:
    """``generate_sea_surface`` sizes and warns by one rule,
    ``sea_surface_n_points``: 8 samples per nominal peak wavelength
    ``2*pi*U**2/g``, at least 500, at most 200 000 (capped with a warning),
    and 500 without a warning for a calm sea (U <= 0.5 m/s), ``g`` the
    standard gravity. At 10 km and 10 m/s that is 1250 points: 1250 is
    quiet, 1249 warns. The winds below
    are the formula's m/s, passed converted to the knots the call takes."""

    @staticmethod
    def _messages(rmax_m, wind, n_points=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            surf = uacpy.generate_sea_surface(
                rmax_m, wind_speed_kn=uacpy.units.ms_to_knots(wind), n_points=n_points,
                rng=np.random.default_rng(0))
        return surf, ' '.join(str(w.message) for w in caught)

    def _warns(self, n_points):
        return 'aliases' in self._messages(10_000.0, 10.0, n_points)[1]

    @staticmethod
    def _rule(rmax_m, wind):
        """ceil(8 * R / lambda_p) + 1, lambda_p = 2*pi*U**2/g."""
        from uacpy.core.constants import STANDARD_GRAVITY_M_S2
        lambda_p = 2 * np.pi * wind ** 2 / STANDARD_GRAVITY_M_S2
        return int(np.ceil(8 * rmax_m / lambda_p)) + 1

    def test_the_documented_rule_is_the_boundary(self):
        n = self._rule(10_000.0, 10.0)
        assert n == 1250
        assert not self._warns(n)
        assert self._warns(n - 1)

    def test_the_default_grid_is_the_rule_and_at_least_500(self):
        surf, text = self._messages(10_000.0, 10.0)
        assert surf.shape[0] == self._rule(10_000.0, 10.0)
        assert 'aliases' not in text
        assert uacpy.generate_sea_surface(
            1000.0, rng=np.random.default_rng(0)).shape[0] == 500

    def test_the_default_grid_is_capped_at_200000_with_a_warning(self):
        from uacpy.core.altimetry import SEA_SURFACE_MAX_POINTS
        from uacpy.core.constants import STANDARD_GRAVITY_M_S2
        lambda_p = 2 * np.pi * 10.0 ** 2 / STANDARD_GRAVITY_M_S2
        under, text = self._messages(199_998.5 * lambda_p / 8, 10.0)
        assert under.shape[0] == SEA_SURFACE_MAX_POINTS and 'cap' not in text
        over, text = self._messages(200_000.5 * lambda_p / 8, 10.0)
        assert over.shape[0] == SEA_SURFACE_MAX_POINTS
        assert 'cap' in text and 'aliases' in text

    def test_a_calm_sea_keeps_500_points_silently(self):
        calm, text = self._messages(10_000.0, 0.5)
        assert calm.shape[0] == 500 and text == ''
        breeze, text = self._messages(10_000.0, 0.51)
        assert breeze.shape[0] == 200_000 and 'cap' in text

    def test_the_range_extent_is_named_rmax_m(self):
        params = inspect.signature(uacpy.generate_sea_surface).parameters
        assert list(params)[0] == 'rmax_m' and 'max_range' not in params
        assert params['n_points'].default is None


class TestTheProfileAnswersItsSoundSpeedAtADepth:
    """``SoundSpeedProfile.sound_speed_at``: the column at ``range`` of a
    range-dependent profile, interpolated in depth by ``method``."""

    @staticmethod
    def _profile():
        return SoundSpeedProfile(depths=[0.0, 100.0],
                                 sound_speed=[[1500.0, 1520.0],
                                              [1480.0, 1500.0]],
                                 ranges=[0.0, 1000.0])

    def test_linear_in_range_and_depth(self):
        c = self._profile().sound_speed_at([50.0], range=500.0)
        assert c.tolist() == [1500.0]

    def test_nearest_picks_the_stored_samples(self):
        c = self._profile().sound_speed_at([60.0], range=900.0,
                                           method='nearest')
        assert c.tolist() == [1500.0]

    def test_a_range_independent_profile_ignores_the_range(self):
        p = SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1480.0)])
        assert p.sound_speed_at([0.0, 100.0], range=1e6).tolist() == [
            1500.0, 1480.0]


class TestFromTemperatureSalinityTakesTheWaterRule:
    """``from_temperature_salinity`` takes each property as a single value,
    an array on ``depths``, or ``(depth, value)`` pairs; with pairs and no
    ``depths`` it builds on the union of their depths."""

    T = [(0.0, 22.0), (60.0, 12.0), (200.0, 10.0)]
    S = [(0.0, 35.5), (300.0, 34.6)]

    def test_pairs_build_on_the_union_of_their_depths(self):
        ssp = SoundSpeedProfile.from_temperature_salinity(
            None, self.T, self.S, formula='mackenzie')
        z = np.array([0.0, 60.0, 200.0, 300.0])
        np.testing.assert_array_equal(ssp.depths, z)
        from uacpy.acoustics import sound_speed_mackenzie
        want = sound_speed_mackenzie(np.interp(z, *np.array(self.T).T),
                                     np.interp(z, *np.array(self.S).T), z)
        np.testing.assert_array_equal(ssp.sound_speed[:, 0], want)

    def test_the_three_forms_mix_on_given_depths(self):
        z = np.array([0.0, 30.0, 100.0])
        mixed = SoundSpeedProfile.from_temperature_salinity(
            z, self.T, 35.0, formula='mackenzie')
        arrays = SoundSpeedProfile.from_temperature_salinity(
            z, np.interp(z, *np.array(self.T).T), np.full(3, 35.0),
            formula='mackenzie')
        np.testing.assert_array_equal(mixed.sound_speed, arrays.sound_speed)

    def test_no_depths_and_no_pairs_is_refused(self):
        with pytest.raises(ConfigurationError, match='no property is given'):
            SoundSpeedProfile.from_temperature_salinity(None, 10.0, 35.0)

    def test_an_array_off_the_depths_is_refused(self):
        with pytest.raises(ConfigurationError, match='must share shape'):
            SoundSpeedProfile.from_temperature_salinity(
                [0.0, 50.0], [10.0, 9.0, 8.0], 35.0)
