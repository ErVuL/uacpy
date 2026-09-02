"""Reference-value anchors for the scalar helpers in
:mod:`uacpy.core.acoustics` — seawater sound speed / density, the bubble
calculators, and ``power_to_db``. All analytic, no binary.

Sources per pin: Mackenzie (1981) validity ranges; Fofonoff EOS-80
one-atmosphere check values; Medwin & Clay eq. (8.2.13) (the Minnaert
breathing frequency); APL-UW TR 9407 eqs. 28a/28b; the worked numbers in
docs/guide/environment.md §5.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.acoustics import (
    bubble_resonance,
    spl,
    bubble_soundspeed,
    bubble_surface_loss,
    density,
    power_to_db,
    soundspeed,
    soundspeed_delgrosso,
    soundspeed_unesco,
)
from uacpy.core.constants import PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER


class TestMackenzieValidityWarnings:
    """``soundspeed`` warns (core/acoustics.py) whenever an input leaves
    Mackenzie's validated ranges — T ∈ [-2, 30] °C, S ∈ [25, 40] PSU,
    D ∈ [0, 8000] m — and stays silent inside them."""

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=35.0),          # T > 30
        dict(temperature=-5.0),          # T < -2
        dict(salinity=10.0),             # S < 25
        dict(salinity=45.0),             # S > 40
        dict(depth=9000.0),              # D > 8000
        dict(depth=-1.0),                # D < 0
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            soundspeed(**kwargs)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = soundspeed()                     # T=27, S=35, D=10
        # The default-point value the environment.md bubble example quotes.
        assert c == pytest.approx(1539.087, abs=1e-3)


class TestUnescoValidityWarnings:
    """``soundspeed_unesco`` announces extrapolation the way ``soundspeed``
    does. Its pressure argument is **decibars** while Chen & Millero state the
    range in bar, so the bound is 10 000 dbar and not 1000: a 5000 m cast is
    comfortably inside it. The cold end warns below −3 °C rather than the
    fit's 0 °C, because seawater is liquid down to about −3 °C under pressure
    and polar deep water lives there."""

    def test_a_deep_cast_in_decibars_is_silent(self):
        # 9000 dbar ≈ 9 km of water — inside 1000 bar, and the value the
        # 10x trap would have flagged.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = soundspeed_unesco(2.0, 34.7, 9000.0)
        assert 1500.0 < c < 1700.0

    def test_pressure_past_the_range_warns_and_names_the_unit(self):
        with pytest.warns(UserWarning, match='DECIBARS'):
            soundspeed_unesco(15.0, 35.0, 10001.0)

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=41.0),          # T > 40
        dict(temperature=-3.5),          # T below the freezing point
        dict(salinity=41.0),             # S > 40
        dict(pressure=-1.0),             # P < 0
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            soundspeed_unesco(**kwargs)

    def test_negative_salinity_is_reported_as_undefined_and_returns_nan(self):
        """Eqn 36's ``B(T,P)·S^1.5`` has no real value below S = 0, so the
        function cannot extrapolate there — it returns NaN. Saying
        "extrapolation" would describe a number the caller never gets, and
        numpy's own "invalid value encountered in power" is suppressed so the
        one diagnostic that names the cause is the one that reaches them."""
        with pytest.warns(UserWarning, match='undefined, not extrapolated'):
            value = soundspeed_unesco(salinity=-1.0)
        assert np.isnan(value)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            soundspeed_unesco(salinity=-1.0)
        assert not [w for w in caught if w.category is RuntimeWarning], (
            [str(w.message) for w in caught])

    def test_polar_deep_water_is_silent(self):
        """The relaxed cold bound exists for this case, and uacpy's own deep
        extrapolation evaluates the formula at exactly −3 °C."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert soundspeed_unesco(-3.0, 34.7, 0.0) == pytest.approx(
                1434.45, abs=0.01)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert soundspeed_unesco() == pytest.approx(1507.0, abs=1.0)


class TestSeawaterDensityEOS80CheckValues:
    """``density`` is the EOS-80 one-atmosphere equation; the canonical
    UNESCO check values pin every coefficient group (pure water, S, S^1.5,
    S² terms)."""

    def test_standard_seawater_check_values(self):
        # UNESCO (1983) / Millero & Poisson one-atmosphere check values.
        assert density(25.0, 35.0) == pytest.approx(1023.343, abs=1e-3)
        assert density(0.0, 35.0) == pytest.approx(1028.106, abs=1e-3)
        # Pure-water limit (S = 0) at 5 °C.
        assert density(5.0, 0.0) == pytest.approx(999.96675, abs=1e-4)


class TestBubbleResonance:
    """``bubble_resonance`` is Medwin & Clay eq. (8.2.13) — the Minnaert
    breathing frequency f = (1/2πa)·√(3γp_A/ρ_A)."""

    def test_one_millimetre_surface_bubble(self):
        # (1/2π·1e-3)·√(3·1.4·1.013e5/1022.476) = 3246.6 Hz — the classic
        # ~3.25 kHz·mm product (fresh-water ρ=1000 gives 3283).
        assert bubble_resonance(1e-3) == pytest.approx(3246.56, abs=0.01)

    def test_frequency_scales_inversely_with_radius(self):
        assert bubble_resonance(1e-4) == pytest.approx(
            10.0 * bubble_resonance(1e-3), rel=1e-12)

    def test_depth_raises_frequency_as_sqrt_ambient_pressure(self):
        # p_A = p0 + ρ g z, so f(z)/f(0) = √(p_A(z)/p0): 1.4106 at 10 m.
        rho, g = 1022.476, 9.80665
        expected = np.sqrt((1.013e5 + rho * g * 10.0) / 1.013e5)
        assert (bubble_resonance(1e-3, depth=10.0) / bubble_resonance(1e-3)
                == pytest.approx(expected, rel=1e-12))


class TestBubbleSoundspeed:
    def test_documented_void_fraction_drop(self):
        """environment.md §5: a void fraction of only 1e-6 drops
        ``bubble_soundspeed`` by 15.5 m/s (1539.1 → 1523.6) at its default
        reference — Wood's equation is that sensitive to entrained gas."""
        c0 = soundspeed()
        c_bubbly = bubble_soundspeed(1e-6)
        assert c_bubbly == pytest.approx(1523.557, abs=1e-3)
        assert c0 - c_bubbly == pytest.approx(15.53, abs=0.01)

    def test_zero_void_fraction_recovers_the_water_speed(self):
        assert bubble_soundspeed(0.0) == pytest.approx(soundspeed(),
                                                       rel=1e-12)


class TestBubbleSurfaceLoss:
    """``bubble_surface_loss`` is APL-UW TR 9407 eqs. 28a/28b:
    SBL = 1.26e-3/sinβ · U^1.57 · f_kHz^0.85 for U ≥ 6 m/s, continued
    exponentially below the 6 m/s breaking-wave threshold. Returns an
    amplitude multiplier in (0, 1], angle in radians."""

    def test_reference_value_at_10ms_20khz_normal_incidence(self):
        # a = 1.26e-3·10^1.57·20^0.85 = 0.598 dB → multiplier 0.9335.
        assert bubble_surface_loss(10.0, 20000.0, 0.0) == pytest.approx(
            0.93354, abs=1e-4)

    def test_multiplier_bounded_and_monotonic_in_wind(self):
        m3 = bubble_surface_loss(3.0, 20000.0, 0.0)
        m10 = bubble_surface_loss(10.0, 20000.0, 0.0)
        assert 0.0 < m10 < m3 <= 1.0

    def test_continuous_across_the_6ms_breaking_wave_threshold(self):
        below = bubble_surface_loss(5.999, 20000.0, 0.0)
        at = bubble_surface_loss(6.0, 20000.0, 0.0)
        assert below == pytest.approx(at, abs=1e-4)

    def test_angle_enters_as_one_over_sin_of_the_grazing_angle(self):
        # angle is incidence in radians; β = π/2 − angle, and the dB loss
        # scales exactly as 1/sin β = 1/cos(angle).
        db0 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 0.0))
        db1 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 1.0))
        assert db1 / db0 == pytest.approx(1.0 / np.cos(1.0), rel=1e-9)


class TestPowerToDb:
    """``power_to_db`` floors ``power`` at :data:`PRESSURE_FLOOR` before the
    log, so a silent sample yields a finite very negative level, never
    ``-inf`` (DOCUMENTATION.md §14)."""

    def test_zero_power_is_finite_at_the_floor_level(self):
        out = power_to_db(0.0)
        assert np.isfinite(out)
        assert out == pytest.approx(
            10.0 * np.log10(PRESSURE_FLOOR / REFERENCE_PRESSURE_WATER ** 2))
        assert out == pytest.approx(-180.0)      # 1e-30 / (1e-6)² = 1e-18

    def test_reference_power_reads_zero_db(self):
        assert power_to_db(REFERENCE_PRESSURE_WATER ** 2) == pytest.approx(0.0)

    def test_custom_floor_is_honoured(self):
        assert power_to_db(0.0, floor=1e-12) == pytest.approx(
            10.0 * np.log10(1e-12 / REFERENCE_PRESSURE_WATER ** 2))


class TestDelGrossoValidityWarnings:
    """``soundspeed_delgrosso`` announces extrapolation the way its two
    siblings do.

    It shipped with no domain guard at all while :func:`soundspeed` and
    :func:`soundspeed_unesco` both had one, so the function its own docstring
    recommends "at high pressure / in deep water" was the one that said
    nothing when handed 50 °C, S = -5 or a pressure ten times its fit.

    The domain is the paper's own: Del Grosso (1974) states "The temperatures
    considered range from 0 to 35 C ... salinity ranges from 29 to 43 ppt ...
    Pressure ranges from 0 to 1000 kg/cm2 gauge", and Etter's Table 2.1
    tabulates the same triple.
    """

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=-3.5),          # T below the coldest seawater
        dict(temperature=35.5),          # T > 35
        dict(salinity=28.0),             # S < 29 — brackish, outside the fit
        dict(salinity=44.0),             # S > 43
        dict(pressure=-1.0),             # P < 0
        dict(pressure=9900.0),           # P > 1000 kg/cm2 == 9806.65 dbar
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            soundspeed_delgrosso(**kwargs)

    def test_a_deep_open_ocean_cast_is_silent(self):
        """The bounds are in the argument's decibars, not the paper's kg/cm2.
        Getting that conversion backwards would warn on every cast past 102 m,
        which is the mistake the UNESCO docstring calls out for its own bar /
        decibar pair."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = soundspeed_delgrosso(2.0, 34.7, 5000.0)
        assert 1500.0 < c < 1600.0

    def test_the_pressure_message_names_the_unit(self):
        with pytest.warns(UserWarning, match='DECIBARS'):
            soundspeed_delgrosso(15.0, 35.0, 9900.0)

    def test_the_salinity_message_points_at_the_equation_that_covers_fresher(self):
        """29 ppt is a floor, not a formality: below it the caller needs a
        different equation, and the message says which."""
        with pytest.warns(UserWarning, match='soundspeed_unesco'):
            soundspeed_delgrosso(salinity=5.0)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert soundspeed_delgrosso() == pytest.approx(1506.67, abs=0.01)

    def test_polar_deep_water_is_silent(self):
        """The cold end is relaxed to −3 °C for the same reason UNESCO's is:
        a literal 0 °C floor fires on every polar and deep cast, and the
        extrapolation across that gap is smooth, monotone, and within Del
        Grosso's own 0.05 m/s standard deviation of UNESCO at −3 °C."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert soundspeed_delgrosso(-3.0, 34.7, 0.0) == pytest.approx(
                1434.51, abs=0.01)


class TestBubbleSurfaceLossAcceptsSequences:
    """``frequency`` and ``angle`` are documented array_like, so a plain
    list must produce the same multipliers as the equivalent ndarray."""

    def test_list_and_ndarray_inputs_agree(self):
        freqs = [10000.0, 20000.0]
        angles = [0.0, 0.3]
        from_lists = bubble_surface_loss(8.0, freqs, angles)
        from_arrays = bubble_surface_loss(
            8.0, np.asarray(freqs), np.asarray(angles))
        np.testing.assert_allclose(from_lists, from_arrays)

    def test_list_inputs_take_the_low_wind_branch_too(self):
        got = bubble_surface_loss(3.0, [10000.0], [0.2])
        want = bubble_surface_loss(3.0, np.array([10000.0]), np.array([0.2]))
        np.testing.assert_allclose(got, want)


class TestSplFloorsSilentSignal:
    """``spl`` floors the rms pressure at ``sqrt(PRESSURE_FLOOR)`` before the
    log, so an all-zero signal returns the same finite level as
    ``power_to_db`` gives zero power, with no runtime warning."""

    def test_all_zero_signal_returns_the_pressure_floor_level(self):
        assert spl(np.zeros(64)) == pytest.approx(
            10.0 * np.log10(PRESSURE_FLOOR))

    def test_zero_signal_level_matches_power_to_db_of_zero_power(self):
        assert spl(np.zeros(64)) == pytest.approx(
            float(power_to_db(0.0, ref=1.0)))

    def test_zero_signal_emits_no_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spl(np.zeros(64))

    def test_nonzero_signal_level_is_the_plain_rms_level(self):
        assert spl(np.full(10, 100.0)) == pytest.approx(40.0)


class TestBubbleSurfaceLossValidatesItsInputs:
    """APL-UW TR 9407 eqs. 28a/28b are written for a non-negative wind speed,
    a positive frequency and an incidence angle inside a quarter turn of the
    surface normal. Outside that the arithmetic answered anyway: a negative
    wind speed took the ``U < 6 m/s`` branch and reported a multiplier of
    ~1.0 (no loss), a negative frequency raised a negative base to 0.85 and
    returned a *complex* multiplier, and an angle past ``π/2`` flipped
    ``sin(β)`` negative and returned a multiplier above 1 — a surface that
    amplifies."""

    @pytest.mark.parametrize('windspeed', [-5.0, -1e-9, float('nan'),
                                           float('inf')])
    def test_a_negative_or_non_finite_windspeed_is_refused(self, windspeed):
        with pytest.raises(ConfigurationError, match='windspeed'):
            bubble_surface_loss(windspeed, 20000.0, 0.0)

    @pytest.mark.parametrize('frequency', [0.0, -1.0, float('nan'),
                                           float('inf')])
    def test_a_non_positive_frequency_is_refused(self, frequency):
        with pytest.raises(ConfigurationError, match='frequency'):
            bubble_surface_loss(10.0, frequency, 0.0)

    def test_one_bad_entry_in_a_frequency_array_is_enough(self):
        with pytest.raises(ConfigurationError, match='frequency'):
            bubble_surface_loss(10.0, np.array([20000.0, -1.0]), 0.0)

    @pytest.mark.parametrize('angle', [1.6, -1.6, np.pi, float('nan')])
    def test_an_angle_outside_a_quarter_turn_is_refused(self, angle):
        with pytest.raises(ConfigurationError, match='angle'):
            bubble_surface_loss(10.0, 20000.0, angle)

    def test_the_angle_message_says_it_is_from_the_normal(self):
        with pytest.raises(ConfigurationError, match='surface normal'):
            bubble_surface_loss(10.0, 20000.0, 2.0)

    def test_exact_grazing_is_the_zero_limit_and_stays_quiet(self):
        """``sin(β) = 0`` is the ``1/sin β → ∞`` limit, whose multiplier is
        0.0. That is a real answer, so it is returned — without the
        divide-by-zero RuntimeWarning it used to raise on the way."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert bubble_surface_loss(10.0, 20000.0, np.pi / 2) == 0.0

    def test_a_grazing_entry_in_an_angle_array_stays_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = bubble_surface_loss(10.0, 20000.0,
                                      np.array([0.0, np.pi / 2]))
        np.testing.assert_allclose(got, [0.93354, 0.0], atol=1e-5)

    def test_a_mirrored_angle_gives_the_same_loss(self):
        """β = π/2 − angle enters only through ``sin β = cos(angle)``, which
        is even, so ±angle are the same ray."""
        assert (bubble_surface_loss(10.0, 20000.0, -1.0)
                == pytest.approx(bubble_surface_loss(10.0, 20000.0, 1.0)))

    def test_the_handbook_values_and_the_secant_angle_law_are_reproduced(self):
        assert bubble_surface_loss(10.0, 20000.0, 0.0) == pytest.approx(
            0.93354, abs=1e-4)
        below = bubble_surface_loss(5.999, 20000.0, 0.0)
        assert below == pytest.approx(
            bubble_surface_loss(6.0, 20000.0, 0.0), abs=1e-4)
        db0 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 0.0))
        db1 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 1.0))
        assert db1 / db0 == pytest.approx(1.0 / np.cos(1.0), rel=1e-9)


class TestArrayCapableHelpersAnnotateArrayReturns:
    """``uacpy`` ships ``py.typed`` (``pyproject.toml``), so every annotation
    in the package is what a downstream type checker sees. A helper annotated
    ``-> float`` that hands back an ``ndarray`` for array input makes the
    checker reject the array call — including the package's own, at
    ``SoundSpeedProfile.from_ts``, which calls ``soundspeed`` on three raveled
    arrays."""

    #: ``(function, array kwargs, scalar kwargs)`` for every helper in
    #: ``core.acoustics`` documented to take either. Both spellings are driven,
    #: so an annotation that admits only one of them fails here.
    CASES = [
        ('soundspeed',
         dict(temperature=np.array([10.0, 20.0]), salinity=35.0, depth=10.0),
         dict(temperature=10.0, salinity=35.0, depth=10.0)),
        ('density',
         dict(temperature=np.array([10.0, 20.0]), salinity=35.0),
         dict(temperature=10.0, salinity=35.0)),
        ('doppler',
         dict(speed=np.array([1.0, 2.0]), frequency=1000.0),
         dict(speed=1.0, frequency=1000.0)),
        ('bubble_resonance',
         dict(radius=np.array([1e-3, 2e-3])), dict(radius=1e-3)),
        ('reflection_coeff',
         dict(angle=np.array([0.2, 0.4]), rho1=2000.0, c1=1800.0,
              rho=1000.0, c=1500.0),
         dict(angle=0.3, rho1=2000.0, c1=1800.0, rho=1000.0, c=1500.0)),
    ]

    @staticmethod
    def _returns_of(name):
        import inspect
        import typing
        from uacpy.core import acoustics
        annotation = inspect.signature(
            getattr(acoustics, name)).return_annotation
        if annotation is inspect.Signature.empty:
            return None
        return set(typing.get_args(annotation)) or {annotation}

    @pytest.mark.parametrize('name,array_kwargs,scalar_kwargs', CASES,
                             ids=[c[0] for c in CASES])
    def test_the_return_annotation_admits_both_shapes(
            self, name, array_kwargs, scalar_kwargs):
        from uacpy.core import acoustics
        function = getattr(acoustics, name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            array_out = function(**array_kwargs)
            scalar_out = function(**scalar_kwargs)
        assert isinstance(array_out, np.ndarray), name
        assert not isinstance(scalar_out, np.ndarray), name

        returns = self._returns_of(name)
        assert returns is not None, f"{name} has no return annotation"
        assert np.ndarray in returns, (
            f"{name} returns an ndarray for array input but annotates "
            f"{returns}")
        assert float in returns, (
            f"{name} returns a scalar for scalar input but annotates "
            f"{returns}")


class TestPressureFloorLevelsMatchTheirDocstrings:
    """``spl`` floors rms pressure in its µPa working unit (silent signal:
    -300 dB re 1 µPa at ``ref=1``); ``power_to_db`` floors ``power`` in
    ``ref**2`` units (silent signal: -180 dB re 1 µPa at its Pa-based
    default). Both docstrings state the split; neither claims the two
    floors coincide at the defaults."""

    def test_spl_floors_a_silent_signal_at_minus_300_db(self):
        assert spl(np.zeros(16)) == pytest.approx(-300.0, rel=1e-12)

    def test_power_to_db_default_ref_floors_at_minus_180_db(self):
        assert float(power_to_db(0.0)) == pytest.approx(-180.0, rel=1e-12)

    def test_the_two_floors_coincide_only_at_unit_ref(self):
        assert float(power_to_db(0.0, ref=1)) == pytest.approx(-300.0,
                                                               rel=1e-12)

    def test_spl_docstring_states_both_floor_levels(self):
        assert '-300 dB' in spl.__doc__
        assert '-180 dB' in spl.__doc__
        assert 'same -300 dB' not in spl.__doc__

    def test_power_to_db_docstring_states_its_own_floor_level(self):
        assert '-180' in power_to_db.__doc__


def test_unesco_reproduces_the_canonical_high_pressure_check_value():
    """Fofonoff & Millard (UNESCO 1983) check value: c = 1731.995 m/s at
    S = 40 PSU, T = 40 °C on the IPTS-68 scale, P = 10000 dbar (1000 bar).
    The temperature argument is ITS-90, so T68 = 40 enters as 40/1.00024."""
    c = soundspeed_unesco(40.0 / 1.00024, 40.0, 10000.0)
    assert float(c) == pytest.approx(1731.995, rel=1e-6)
