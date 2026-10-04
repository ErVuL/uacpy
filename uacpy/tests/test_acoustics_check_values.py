"""Published check values for :mod:`uacpy.core.acoustics`.

Every formula in the package is anchored to a number someone else published:
the ``seawater`` sound-speed equations and density, the ``bubbles``
calculators, and ``levels``' volts-to-pascals-to-dB chain. All analytic, no
binary.

Sources per pin: Mackenzie (1981) validity ranges; Fofonoff EOS-80
one-atmosphere check values; Medwin & Clay eq. (8.2.13) (the Minnaert
breathing frequency); APL-UW TR 9407 eqs. 28a/28b; the worked numbers in
docs/guide/environment.md §5.
"""

import warnings

import numpy as np
import pytest

# bubble_surface_loss takes knots; the handbook's winds below are its m/s
from uacpy.core.units import ms_to_knots

from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.acoustics import (
    bubble_resonance,
    pressure,
    spl,
    bubble_sound_speed,
    bubble_surface_loss,
    density,
    power_to_dB,
    sound_speed_mackenzie,
    sound_speed_delgrosso,
    sound_speed_teos10,
    sound_speed_unesco,
)
from uacpy.core.constants import PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER
from uacpy.tests.conftest import recorded_warnings


class TestMackenzieValidityWarnings:
    """``sound_speed_mackenzie`` warns (core/acoustics/seawater.py) whenever an input leaves
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
            sound_speed_mackenzie(**kwargs)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = sound_speed_mackenzie()                     # T=10, S=35, D=0
        # The package's reference water (10 °C, 35 psu, surface).
        assert c == pytest.approx(1489.803, abs=1e-3)


class TestUnescoValidityWarnings:
    """``sound_speed_unesco`` announces extrapolation the way ``sound_speed_mackenzie``
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
            c = sound_speed_unesco(2.0, 34.7, pressure_dbar=9000.0)
        assert 1500.0 < c < 1700.0

    def test_pressure_past_the_range_warns_and_names_the_unit(self):
        with pytest.warns(UserWarning, match='DECIBARS'):
            sound_speed_unesco(15.0, 35.0, pressure_dbar=10001.0)

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=41.0),          # T > 40
        dict(temperature=-3.5),          # T below the freezing point
        dict(salinity=41.0),             # S > 40
        dict(pressure_dbar=-1.0),             # P < 0
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            sound_speed_unesco(**kwargs)

    def test_negative_salinity_is_reported_as_undefined_and_returns_nan(self):
        """Eqn 36's ``B(T,P)·S^1.5`` has no real value below S = 0, so the
        function cannot extrapolate there — it returns NaN. Saying
        "extrapolation" would describe a number the caller never gets, and
        numpy's own "invalid value encountered in power" is suppressed so the
        one diagnostic that names the cause is the one that reaches them."""
        with pytest.warns(UserWarning, match='undefined, not extrapolated'):
            value = sound_speed_unesco(salinity=-1.0)
        assert np.isnan(value)

        with recorded_warnings() as caught:
            sound_speed_unesco(salinity=-1.0)
        assert not [w for w in caught if w.category is RuntimeWarning], (
            [str(w.message) for w in caught])

    def test_polar_deep_water_is_silent(self):
        """The relaxed cold bound exists for this case, and uacpy's own deep
        extrapolation evaluates the formula at exactly −3 °C."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert sound_speed_unesco(-3.0, 34.7, pressure_dbar=0.0) == pytest.approx(
                1434.45, abs=0.01)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert sound_speed_unesco() == pytest.approx(1489.83, abs=0.01)


class TestSeawaterDensityEOS80CheckValues:
    """``density`` is the EOS-80 one-atmosphere equation; the canonical
    UNESCO check values pin every coefficient group (pure water, S, S^1.5,
    S² terms)."""

    def test_standard_seawater_check_values(self):
        # UNESCO (1983) / Millero & Poisson one-atmosphere check values. They
        # are stated at IPTS-68 temperatures and ``density`` takes ITS-90, so
        # each is called at t90 = t68 / 1.00024.
        t90 = 1.0 / 1.00024
        assert density(25.0 * t90, 35.0) == pytest.approx(1023.343, abs=1e-3)
        assert density(0.0, 35.0) == pytest.approx(1028.106, abs=1e-3)
        # Pure-water limit (S = 0) at 5 °C.
        assert density(5.0 * t90, 0.0) == pytest.approx(999.96675, abs=1e-4)


class TestBubbleResonance:
    """``bubble_resonance`` is Medwin & Clay eq. (8.2.13) — the Minnaert
    breathing frequency f = (1/2πa)·√(3γp_A/ρ_A)."""

    def test_one_millimetre_surface_bubble(self):
        # (1/2π·1e-3)·√(3·1.4·101325/1027) = 3239.8 Hz — the classic
        # ~3.25 kHz·mm product (fresh-water ρ=1000 gives 3283), in the
        # package's one water density.
        assert bubble_resonance(1e-3) == pytest.approx(3239.80, abs=0.01)

    def test_frequency_scales_inversely_with_radius(self):
        assert bubble_resonance(1e-4) == pytest.approx(
            10.0 * bubble_resonance(1e-3), rel=1e-12)

    def test_water_density_is_a_kg_per_m3_keyword(self):
        # Fresh water at 1000 kg/m³ gives the 3283 Hz of the comment above.
        assert bubble_resonance(1e-3, water_density_kg_m3=1000.0) == \
            pytest.approx(3283.0, abs=0.5)

    def test_depth_raises_frequency_as_sqrt_ambient_pressure(self):
        # p_A = p0 + ρ g z, so f(z)/f(0) = √(p_A(z)/p0): 1.4106 at 10 m.
        rho, g = 1027.0, 9.80665
        expected = np.sqrt((101325.0 + rho * g * 10.0) / 101325.0)
        assert (bubble_resonance(1e-3, depth=10.0) / bubble_resonance(1e-3)
                == pytest.approx(expected, rel=1e-12))


class TestBubbleSoundspeed:
    def test_documented_void_fraction_drop(self):
        """environment.md §5: a void fraction of only 1e-6 drops
        ``bubble_sound_speed`` by 14.4 m/s (1500.0 → 1485.6) at its default,
        the nominal 1500 m/s — Wood's equation is that sensitive to entrained
        gas."""
        c_bubbly = bubble_sound_speed(1e-6)
        assert c_bubbly == pytest.approx(1485.613, abs=1e-3)
        assert DEFAULT_SOUND_SPEED - c_bubbly == pytest.approx(14.39, abs=0.01)

    def test_zero_void_fraction_recovers_the_water_speed(self):
        assert bubble_sound_speed(0.0) == pytest.approx(DEFAULT_SOUND_SPEED,
                                                       rel=1e-12)


class TestBubbleSurfaceLoss:
    """``bubble_surface_loss`` is APL-UW TR 9407 eqs. 28a/28b:
    SBL = 1.26e-3/sinβ · U^1.57 · f_kHz^0.85 for U ≥ 6 m/s, continued
    exponentially below the 6 m/s breaking-wave threshold. Returns an
    amplitude multiplier in (0, 1], grazing angle in degrees."""

    def test_reference_value_at_10ms_20khz_normal_incidence(self):
        # a = 1.26e-3·10^1.57·20^0.85 = 0.598 dB → multiplier 0.9335.
        assert bubble_surface_loss(ms_to_knots(10.0), 20000.0, grazing_deg=90.0) == \
            pytest.approx(0.93354, abs=1e-4)

    def test_multiplier_bounded_and_monotonic_in_wind(self):
        m3 = bubble_surface_loss(ms_to_knots(3.0), 20000.0, 90.0)
        m10 = bubble_surface_loss(ms_to_knots(10.0), 20000.0, 90.0)
        assert 0.0 < m10 < m3 <= 1.0

    def test_continuous_across_the_6ms_breaking_wave_threshold(self):
        below = bubble_surface_loss(ms_to_knots(5.999), 20000.0, 90.0)
        at = bubble_surface_loss(ms_to_knots(6.0), 20000.0, 90.0)
        assert below == pytest.approx(at, abs=1e-4)

    def test_grazing_angle_enters_as_one_over_its_sine(self):
        # The dB loss scales exactly as 1/sin(grazing): twice the normal-
        # incidence loss at 30 degrees.
        db90 = -20.0 * np.log10(bubble_surface_loss(ms_to_knots(10.0), 20000.0, 90.0))
        db30 = -20.0 * np.log10(bubble_surface_loss(ms_to_knots(10.0), 20000.0, 30.0))
        assert db30 / db90 == pytest.approx(2.0, rel=1e-9)

    def test_the_angle_is_grazing_not_from_the_normal(self):
        # A shallow ray loses more than a steep one; a from-the-normal
        # reading of the argument would give the reverse order.
        steep = bubble_surface_loss(ms_to_knots(10.0), 20000.0, 80.0)
        shallow = bubble_surface_loss(ms_to_knots(10.0), 20000.0, 10.0)
        assert shallow < steep


class TestPowerToDb:
    """``power_to_dB`` floors ``power`` at :data:`PRESSURE_FLOOR` before the
    log, so a silent sample yields a finite very negative level, never
    ``-inf`` (DOCUMENTATION.md §14)."""

    def test_zero_power_is_finite_at_the_floor_level(self):
        out = power_to_dB(0.0)
        assert np.isfinite(out)
        assert out == pytest.approx(
            10.0 * np.log10(PRESSURE_FLOOR / REFERENCE_PRESSURE_WATER ** 2))
        assert out == pytest.approx(-180.0)      # 1e-30 / (1e-6)² = 1e-18

    def test_reference_power_reads_zero_dB(self):
        assert power_to_dB(REFERENCE_PRESSURE_WATER ** 2) == pytest.approx(0.0)

    def test_custom_floor_is_honoured(self):
        assert power_to_dB(0.0, floor=1e-12) == pytest.approx(
            10.0 * np.log10(1e-12 / REFERENCE_PRESSURE_WATER ** 2))


class TestDelGrossoValidityWarnings:
    """``sound_speed_delgrosso`` announces extrapolation the way its two
    siblings do.

    It shipped with no domain guard at all while :func:`sound_speed_mackenzie` and
    :func:`sound_speed_unesco` both had one, so the function its own docstring
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
        dict(pressure_dbar=-1.0),             # P < 0
        dict(pressure_dbar=9900.0),           # P > 1000 kg/cm2 == 9806.65 dbar
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            sound_speed_delgrosso(**kwargs)

    def test_a_deep_open_ocean_cast_is_silent(self):
        """The bounds are in the argument's decibars, not the paper's kg/cm2.
        Getting that conversion backwards would warn on every cast past 102 m,
        which is the mistake the UNESCO docstring calls out for its own bar /
        decibar pair."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = sound_speed_delgrosso(2.0, 34.7, pressure_dbar=5000.0)
        assert 1500.0 < c < 1600.0

    def test_the_pressure_message_names_the_unit(self):
        with pytest.warns(UserWarning, match='DECIBARS'):
            sound_speed_delgrosso(15.0, 35.0, pressure_dbar=9900.0)

    def test_the_salinity_message_points_at_the_equation_that_covers_fresher(self):
        """29 ppt is a floor, not a formality: below it the caller needs a
        different equation, and the message says which."""
        with pytest.warns(UserWarning, match='sound_speed_unesco'):
            sound_speed_delgrosso(salinity=5.0)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert sound_speed_delgrosso() == pytest.approx(1489.78, abs=0.01)

    def test_polar_deep_water_is_silent(self):
        """The cold end is relaxed to −3 °C for the same reason UNESCO's is:
        a literal 0 °C floor fires on every polar and deep cast, and the
        extrapolation across that gap is smooth, monotone, and within Del
        Grosso's own 0.05 m/s standard deviation of UNESCO at −3 °C."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert sound_speed_delgrosso(-3.0, 34.7, pressure_dbar=0.0) == pytest.approx(
                1434.51, abs=0.01)


class TestBubbleSurfaceLossAcceptsSequences:
    """``frequency`` and ``grazing_deg`` are documented array_like, so a plain
    list must produce the same multipliers as the equivalent ndarray."""

    def test_list_and_ndarray_inputs_agree(self):
        freqs = [10000.0, 20000.0]
        angles = [90.0, 72.8]
        from_lists = bubble_surface_loss(ms_to_knots(8.0), freqs, angles)
        from_arrays = bubble_surface_loss(ms_to_knots(8.0), np.asarray(freqs), np.asarray(angles))
        np.testing.assert_allclose(from_lists, from_arrays)

    def test_list_inputs_take_the_low_wind_branch_too(self):
        got = bubble_surface_loss(ms_to_knots(3.0), [10000.0], [78.5])
        want = bubble_surface_loss(ms_to_knots(3.0), np.array([10000.0]), np.array([78.5]))
        np.testing.assert_allclose(got, want)


class TestSplFloorsSilentSignal:
    """``spl`` floors the rms pressure at ``sqrt(PRESSURE_FLOOR)`` before the
    log, so an all-zero signal returns a finite level and no runtime warning.

    Both this and ``power_to_dB`` work in pascals against the same default
    reference, so the level they give a silent signal is the same number
    without either being told a reference — which is the point of the module
    speaking one unit.
    """

    def test_all_zero_signal_returns_the_pressure_floor_level(self):
        assert spl(np.zeros(64)) == pytest.approx(
            20.0 * np.log10(np.sqrt(PRESSURE_FLOOR)
                            / REFERENCE_PRESSURE_WATER))

    def test_zero_signal_level_matches_power_to_dB_of_zero_power(self):
        assert spl(np.zeros(64)) == pytest.approx(float(power_to_dB(0.0)))

    def test_zero_signal_emits_no_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spl(np.zeros(64))

    def test_nonzero_signal_level_is_the_plain_rms_level(self):
        # 100 µPa rms, written in the pascals this module reads.
        assert spl(np.full(10, 100e-6)) == pytest.approx(40.0)


class TestPressureCalibratesVoltsToPascals:
    """``pressure`` is where a recording's own unit enters the package: it
    divides the signal by the hydrophone sensitivity, which data sheets quote
    in dB re 1 V/µPa, and returns **pascals** — the unit every level default
    downstream reads.

    Returning µPa instead reads 120 dB high through ``spl``, ``power_to_dB``
    and every plotter, with nothing in the numbers saying so. Nothing in the
    standing suite runs a ``core.acoustics`` doctest (only ``io`` writers get
    theirs executed, in ``test_documentation``), so the unit is pinned here.
    """

    #: A mid-range hydrophone data sheet: SH in dB re 1 V/µPa, preamp gain dB.
    SENSITIVITY, GAIN = -165.0, 20.0

    def _micropascals_per_volt(self):
        return 1.0 / (10.0 ** (self.SENSITIVITY / 20.0)
                      * 10.0 ** (self.GAIN / 20.0))

    def test_unity_scale_factors_return_the_voltage_in_pascals(self):
        got = pressure(np.array([0.0, 0.5, -0.5]), sensitivity=0, gain=0)
        np.testing.assert_allclose(got, [0.0, 0.5e-6, -0.5e-6])

    def test_a_data_sheet_sensitivity_lands_in_pascals(self):
        got = pressure(np.array([1e-3]), sensitivity=self.SENSITIVITY,
                       gain=self.GAIN)
        assert float(got[0]) == pytest.approx(
            1e-6 * 1e-3 * self._micropascals_per_volt(), rel=1e-12)
        assert float(got[0]) == pytest.approx(1.77828e-2, rel=1e-5)

    def test_adc_counts_take_the_same_pascal_scale_as_volts(self):
        # Half of full scale on a signed 16-bit sample against a 1 V
        # reference is 0.5 V, and the level it reaches is stated in pascals
        # rather than only against the volt call, which would hold whatever
        # unit the function returned.
        counts = pressure(np.array([16384]), sensitivity=self.SENSITIVITY,
                          gain=self.GAIN, volt_params=(16, 1.0))
        volts = pressure(np.array([0.5]), sensitivity=self.SENSITIVITY,
                         gain=self.GAIN)
        np.testing.assert_allclose(counts, volts)
        assert float(counts[0]) == pytest.approx(
            1e-6 * 0.5 * self._micropascals_per_volt(), rel=1e-12)

    def test_the_chain_to_spl_needs_no_reference_argument(self):
        # dB re 1 µPa is 20*log10 of the µPa count, so the expected level
        # carries no pascal factor: a return in µPa would read 120 dB high.
        volts = np.full(256, 1e-3)
        level = spl(pressure(volts, sensitivity=self.SENSITIVITY,
                             gain=self.GAIN))
        assert level == pytest.approx(
            20.0 * np.log10(1e-3 * self._micropascals_per_volt()), rel=1e-12)


class TestBubbleSurfaceLossValidatesItsInputs:
    """APL-UW TR 9407 eqs. 28a/28b are written for a non-negative wind speed,
    a positive frequency and a grazing angle in ``[0, 90]`` degrees. Outside
    that the arithmetic answers anyway: a negative wind speed takes the
    ``U < 6 m/s`` branch and reports a multiplier of ~1.0 (no loss), a
    negative frequency raises a negative base to 0.85 and returns a
    *complex* multiplier, and a negative grazing angle turns ``sin``
    negative and returns a multiplier above 1 — a surface that amplifies."""

    @pytest.mark.parametrize('wind_speed_kn', [-5.0, -1e-9, float('nan'),
                                                float('inf')])
    def test_a_negative_or_non_finite_wind_speed_is_refused(self,
                                                            wind_speed_kn):
        with pytest.raises(ConfigurationError, match='wind_speed_kn'):
            bubble_surface_loss(wind_speed_kn, 20000.0, 90.0)

    def test_the_bubble_helpers_name_their_speeds_with_the_package_words(self):
        import inspect
        from uacpy.core.acoustics import bubble_sound_speed
        surface = inspect.signature(bubble_surface_loss).parameters
        mixture = inspect.signature(bubble_sound_speed).parameters
        assert 'wind_speed_kn' in surface and 'windspeed' not in surface
        assert 'grazing_deg' in surface and 'angle' not in surface
        assert 'sound_speed' in mixture and 'c' not in mixture
        assert bubble_sound_speed(1e-5, sound_speed=1500.0) == \
            bubble_sound_speed(1e-5)

    @pytest.mark.parametrize('frequency', [0.0, -1.0, float('nan'),
                                           float('inf')])
    def test_a_non_positive_frequency_is_refused(self, frequency):
        with pytest.raises(ConfigurationError, match='frequency'):
            bubble_surface_loss(ms_to_knots(10.0), frequency, 90.0)

    def test_one_bad_entry_in_a_frequency_array_is_enough(self):
        with pytest.raises(ConfigurationError, match='frequency'):
            bubble_surface_loss(ms_to_knots(10.0), np.array([20000.0, -1.0]), 90.0)

    @pytest.mark.parametrize('grazing', [-1e-9, -1.0, 90.0 + 1e-9, 180.0,
                                         float('nan')])
    def test_a_grazing_angle_outside_0_to_90_is_refused(self, grazing):
        with pytest.raises(ConfigurationError, match='grazing_deg'):
            bubble_surface_loss(ms_to_knots(10.0), 20000.0, grazing)

    @pytest.mark.parametrize('grazing', [0.0, 90.0])
    def test_both_ends_of_the_grazing_range_are_accepted(self, grazing):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert 0.0 <= bubble_surface_loss(ms_to_knots(10.0), 20000.0, grazing) <= 1.0

    def test_zero_grazing_is_the_zero_limit_and_stays_quiet(self):
        """``sin(0) = 0`` is the ``1/sin → ∞`` limit, whose multiplier is
        0.0. That is a real answer, so it is returned without a
        divide-by-zero RuntimeWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert bubble_surface_loss(ms_to_knots(10.0), 20000.0, 0.0) == 0.0

    def test_a_zero_grazing_entry_in_an_angle_array_stays_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = bubble_surface_loss(ms_to_knots(10.0), 20000.0, np.array([90.0, 0.0]))
        np.testing.assert_allclose(got, [0.93354, 0.0], atol=1e-5)

    def test_the_handbook_values_and_the_cosecant_angle_law_are_reproduced(self):
        assert bubble_surface_loss(ms_to_knots(10.0), 20000.0, 90.0) == pytest.approx(
            0.93354, abs=1e-4)
        below = bubble_surface_loss(ms_to_knots(5.999), 20000.0, 90.0)
        assert below == pytest.approx(
            bubble_surface_loss(ms_to_knots(6.0), 20000.0, 90.0), abs=1e-4)
        db90 = -20.0 * np.log10(bubble_surface_loss(ms_to_knots(10.0), 20000.0, 90.0))
        db20 = -20.0 * np.log10(bubble_surface_loss(ms_to_knots(10.0), 20000.0, 20.0))
        assert db20 / db90 == pytest.approx(1.0 / np.sin(np.radians(20.0)),
                                            rel=1e-9)


class TestArrayCapableHelpersAnnotateArrayReturns:
    """``uacpy`` ships ``py.typed`` (``pyproject.toml``), so every annotation
    in the package is what a downstream type checker sees. A helper annotated
    ``-> float`` that hands back an ``ndarray`` for array input makes the
    checker reject the array call — including the package's own, at
    ``SoundSpeedProfile.from_temperature_salinity``, which calls ``sound_speed_mackenzie`` on three
    raveled arrays."""

    #: ``(function, array kwargs, scalar kwargs)`` for every helper in
    #: ``core.acoustics`` documented to take either. Both spellings are driven,
    #: so an annotation that admits only one of them fails here.
    CASES = [
        ('sound_speed_mackenzie',
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
         dict(grazing_deg=np.array([20.0, 40.0]), sound_speed=1800.0,
              density=2.0),
         dict(grazing_deg=30.0, sound_speed=1800.0, density=2.0)),
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
    """``spl`` floors rms pressure at ``sqrt(PRESSURE_FLOOR)`` pascals and
    ``power_to_dB`` floors ``power`` at ``PRESSURE_FLOOR`` Pa², which is the
    same floor on either side of the square. Both read pascals against the
    same default reference, so a silent signal takes the same -180 dB re
    1 µPa through either door, and both docstrings state that level."""

    def test_spl_floors_a_silent_signal_at_minus_180_dB(self):
        assert spl(np.zeros(16)) == pytest.approx(-180.0, rel=1e-12)

    def test_power_to_dB_default_ref_floors_at_minus_180_dB(self):
        assert float(power_to_dB(0.0)) == pytest.approx(-180.0, rel=1e-12)

    def test_the_two_floors_coincide_at_the_default_reference(self):
        assert spl(np.zeros(16)) == pytest.approx(float(power_to_dB(0.0)),
                                                  rel=1e-12)

    def test_spl_docstring_states_the_floor_level(self):
        assert '-180 dB' in spl.__doc__

    def test_power_to_dB_docstring_states_its_own_floor_level(self):
        assert '-180' in power_to_dB.__doc__


def test_unesco_reproduces_the_canonical_high_pressure_check_value():
    """Fofonoff & Millard (UNESCO 1983) check value: c = 1731.995 m/s at
    S = 40 PSU, T = 40 °C on the IPTS-68 scale, P = 10000 dbar (1000 bar).
    The temperature argument is ITS-90, so T68 = 40 enters as 40/1.00024."""
    c = sound_speed_unesco(40.0 / 1.00024, 40.0, pressure_dbar=10000.0)
    assert float(c) == pytest.approx(1731.995, rel=1e-6)


class TestTeos10SoundSpeedEvaluatesTheGibbsFunction:
    """``sound_speed_teos10`` is Eqn. (2.17.1) of the TEOS-10 manual (IOC
    Manuals and Guides 56, p. 22), ``c = g_P·sqrt(g_TT / (g_TP² − g_TT·g_PP))``,
    evaluated on the IAPWS-09 pure-water plus IAPWS-08 saline Gibbs function
    whose coefficients the manual tabulates in appendices G and H. It takes
    the same ``(ITS-90 °C, Practical Salinity, dbar)`` triple as the UNESCO
    and Del Grosso equations and converts Practical to Reference Salinity
    (``× 35.16504/35``) internally.

    Reference values: GSW-Python 3.6.23 ``gsw.sound_speed_t_exact(SA, t, p)``
    with ``SA = SP × 35.16504/35`` — the TEOS-10 toolbox's own evaluation of
    the same Gibbs function. Agreement to 1e-5 m/s pins every coefficient:
    a single wrong digit in either table moves the result by far more.
    """

    @pytest.mark.parametrize('temperature, salinity, pressure, expected', [
        (15.0, 35.0, 0.0, 1506.673601),
        (0.0, 35.0, 0.0, 1449.024607),       # the Standard Ocean point
        (25.0, 35.0, 0.0, 1534.357131),
        (30.0, 40.0, 0.0, 1550.694855),
        (2.0, 34.7, 5000.0, 1541.614915),
        (4.0, 35.0, 4000.0, 1533.099822),
        (1.5, 34.7, 6000.0, 1557.008080),
        (4.0, 35.0, 10000.0, 1637.285857),
        (-3.0, 34.7, 0.0, 1434.333863),
    ])
    def test_matches_the_gsw_reference_to_ten_micrometres_per_second(
            self, temperature, salinity, pressure, expected):
        assert sound_speed_teos10(temperature, salinity, pressure_dbar=pressure) == \
            pytest.approx(expected, abs=1e-5)

    def test_fresh_water_is_finite_where_the_saline_term_has_x_squared_ln_x(self):
        """At ``S = 0`` the saline Gibbs function's ``x²·ln x`` terms are the
        limit 0, not ``0 × (−inf) = nan``."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = sound_speed_teos10(10.0, 0.0, pressure_dbar=0.0)
        assert c == pytest.approx(1447.284153, abs=1e-5)

    def test_returns_a_python_float_for_scalars_and_broadcasts_arrays(self):
        assert isinstance(sound_speed_teos10(15.0, 35.0, pressure_dbar=0.0), float)
        t = np.array([0.0, 15.0, 25.0])
        p = np.array([[0.0], [4000.0]])
        c = sound_speed_teos10(t, 35.0, pressure_dbar=p)
        assert c.shape == (2, 3)
        assert c[0, 1] == pytest.approx(1506.673601, abs=1e-5)
        assert c[1, 0] == pytest.approx(
            sound_speed_teos10(0.0, 35.0, pressure_dbar=4000.0), abs=1e-9)

    def test_sits_with_del_grosso_not_unesco_in_deep_water(self):
        """The Feistel (2008) Gibbs function was fitted to sound-speed data
        (manual appendix O, Table O.1; rms 0.035 m/s), so at depth it
        reproduces Del Grosso and exposes the ~0.6 m/s pressure bias of the
        uncorrected Chen–Millero polynomial (APL-UW TR 9407, "Chen-Millero-Li
        Equation"). Measured 2026-09-07 at the audit's deep fixture point."""
        t, s, p = 2.0, 34.7, 5000.0
        c = sound_speed_teos10(t, s, pressure_dbar=p)
        assert abs(c - sound_speed_delgrosso(t, s, pressure_dbar=p)) < 0.05
        assert sound_speed_unesco(t, s, pressure_dbar=p) - c > 0.5


class TestTeos10ValidityWarnings:
    """``sound_speed_teos10`` announces extrapolation the way its siblings do.

    The domain is the manual's own (§2.6): the saline Gibbs function "is
    valid over the ranges 0 < S_A < 42 g/kg, −6.0 °C < t < 40 °C, and
    0 < p < 10⁴ dbar". 42 g/kg of Absolute Salinity is 41.80 on the
    Practical scale this argument takes.
    """

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=-6.5),          # T < -6
        dict(temperature=40.5),          # T > 40
        dict(salinity=42.5),             # S > 41.80 PSU (42 g/kg)
        dict(pressure_dbar=-1.0),             # P < 0
        dict(pressure_dbar=10100.0),          # P > 10000 dbar
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            sound_speed_teos10(**kwargs)

    def test_negative_salinity_is_undefined_not_extrapolated(self):
        """``x = sqrt(S_A / S_u)`` has no real value below zero; the result is
        NaN and the message says so, as UNESCO's does for its ``S^1.5``."""
        with pytest.warns(UserWarning, match='undefined'):
            c = sound_speed_teos10(10.0, -1.0, pressure_dbar=0.0)
        assert np.isnan(c)

    def test_the_pressure_message_names_the_unit(self):
        with pytest.warns(UserWarning, match='DECIBARS'):
            sound_speed_teos10(15.0, 35.0, pressure_dbar=10100.0)

    def test_a_deep_polar_cast_and_the_defaults_are_silent(self):
        """−3 °C sits inside this equation's own −6 °C floor, so unlike the
        two older fits nothing here is relaxed."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            assert 1400.0 < sound_speed_teos10(-3.0, 34.7, pressure_dbar=5000.0) < 1600.0
            assert sound_speed_teos10() == pytest.approx(1489.79, abs=0.01)



class TestEveryDefaultDescribesOneReferenceWater:
    """Every seawater default is one state: 10 °C, 35 psu, the surface.

    Before, Mackenzie and ``density`` defaulted to 27 °C, the other three
    sound-speed equations to 15 °C and the Francois-Garrison helper to 10 °C
    at 1000 m, so a default density, sound speed and bubble resonance
    described three different waters. The pins below are between the
    functions, so a default that drifts off the shared state fails here."""

    def test_the_density_default_is_the_package_water_density(self):
        assert density() == pytest.approx(
            DEFAULT_WATER_DENSITY_G_CM3 * 1000.0, abs=0.05)

    def test_the_four_sound_speed_equations_agree_at_their_defaults(self):
        speeds = [sound_speed_mackenzie(), sound_speed_unesco(),
                  sound_speed_delgrosso(), sound_speed_teos10()]
        assert max(speeds) - min(speeds) < 0.1

    def test_standalone_water_defaults_read_the_package_constants(self):
        from uacpy.core.acoustics import doppler, reflection_coeff
        assert doppler(15.0, 1000.0) == pytest.approx(
            1000.0 * (1 + 15.0 / DEFAULT_SOUND_SPEED), rel=1e-15)
        angles = np.linspace(0.0, 90.0, 7)
        np.testing.assert_allclose(
            reflection_coeff(angles, sound_speed=1700.0, density=1.8),
            reflection_coeff(angles, sound_speed=1700.0, density=1.8,
                             water_density=DEFAULT_WATER_DENSITY_G_CM3,
                             water_sound_speed=DEFAULT_SOUND_SPEED),
            rtol=0, atol=0)

    def test_the_source_array_methods_default_to_the_package_sound_speed(self):
        import inspect
        from uacpy import Source
        for method in (Source.array_factor, Source.array_beam_pattern):
            default = inspect.signature(method).parameters['sound_speed'].default
            assert default is DEFAULT_SOUND_SPEED, method.__name__


class TestAnalyticReferenceFields:
    """uacpy.analytic against its own closed forms."""

    @staticmethod
    def _geometry():
        from uacpy import Receiver, Source
        return (Source(depths=[36.0], frequencies=[50.0]),
                Receiver(depths=np.linspace(5, 95, 10),
                         ranges=np.linspace(200, 1000, 9)))

    def test_free_field_is_spherical_spreading(self):
        from uacpy import analytic
        src, rx = self._geometry()
        f = analytic.free_field(src, rx)
        big_r = np.hypot(rx.depths[:, None] - 36.0, rx.ranges[None, :])
        np.testing.assert_allclose(f.tl, 20 * np.log10(big_r), atol=1e-10)
        assert f.phase_reference == 'travelling_wave'
        # Travelling-wave carrier exp(-ikR): phase decreases with distance.
        k = 2 * np.pi * 50.0 / DEFAULT_SOUND_SPEED
        np.testing.assert_allclose(f.data * big_r, np.exp(-1j * k * big_r))

    def test_lloyd_mirror_vanishes_at_the_surface(self):
        from uacpy import Receiver, analytic
        src, _ = self._geometry()
        f = analytic.lloyd_mirror(src, Receiver(depths=[0.0, 10.0],
                                                ranges=[500.0]))
        assert abs(f.data[0, 0]) < 1e-15
        assert abs(f.data[1, 0]) > 1e-4

    def test_ideal_waveguide_modal_sum_equals_its_image_sum(self):
        from uacpy import Environment, analytic
        from uacpy.core.boundary import BoundaryProperties
        src, rx = self._geometry()
        env = Environment(bathymetry=100, ssp=1500,
                          bottom=BoundaryProperties(acoustic_type='vacuum'))
        modal = analytic.ideal_waveguide(env, src, rx).data
        k = 2 * np.pi * 50.0 / 1500.0
        z, r = rx.depths[:, None], rx.ranges[None, :]
        images = 0
        for n in range(-3000, 3001):
            for sign, zi in ((1, 200.0 * n + 36.0), (-1, 200.0 * n - 36.0)):
                dist = np.hypot(z - zi, r)
                images = images + sign * np.exp(-1j * k * dist) / dist
        # The residual is the truncated image sum's, not the modal sum's.
        assert np.abs(modal - images).max() < 1e-3 * np.abs(images).max()

    def test_closed_forms_refuse_what_they_do_not_model(self):
        from uacpy import Environment, analytic
        from uacpy.core.boundary import BoundaryProperties
        src, rx = self._geometry()
        lossy = Environment(bathymetry=100, ssp=1500, bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1800, density=2.0,
            attenuation=0.5))
        with pytest.raises(ConfigurationError, match="lossless"):
            analytic.pekeris(lossy, src, rx)
        with pytest.raises(ConfigurationError, match="rigid or vacuum"):
            analytic.ideal_waveguide(lossy, src, rx)
        refracting = Environment(bathymetry=100, ssp=[(0, 1500), (100, 1480)],
                                 bottom=BoundaryProperties(acoustic_type='rigid'))
        with pytest.raises(ConfigurationError, match="isovelocity"):
            analytic.ideal_waveguide(refracting, src, rx)

    def test_sediment_layers_over_a_rigid_basement_are_refused(self):
        from uacpy import Environment, analytic
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import SeabedColumn
        src, rx = self._geometry()
        layered = Environment(bathymetry=100, ssp=1500, bottom=SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.2)],
            halfspace=BoundaryProperties(acoustic_type='rigid')))
        with pytest.raises(ConfigurationError, match="sediment layers"):
            analytic.ideal_waveguide(layered, src, rx)

    @pytest.mark.parametrize('closed_form', ['ideal_waveguide', 'pekeris'])
    def test_a_rough_surface_is_refused(self, closed_form):
        from uacpy import Environment, analytic
        from uacpy.core.boundary import BoundaryProperties
        src, rx = self._geometry()
        bottom = (BoundaryProperties(acoustic_type='rigid')
                  if closed_form == 'ideal_waveguide' else
                  BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1700.0, density=1.5,
                                     attenuation=0.0))
        rough = Environment(bathymetry=100, ssp=1500, bottom=bottom,
                            altimetry=[(0.0, 0.0), (1000.0, 0.5)])
        with pytest.raises(ConfigurationError, match="altimetry"):
            getattr(analytic, closed_form)(rough, src, rx)
        flat = Environment(bathymetry=100, ssp=1500, bottom=bottom,
                           altimetry=[(0.0, 0.0), (1000.0, 0.0)])
        assert np.isfinite(getattr(analytic, closed_form)(flat, src, rx).tl).all()


class TestTheIdealWaveguideNamesItsCutoffResonances:
    """At a mode's cutoff ``k_r = 0`` and ``H_0(k_r r)`` is infinite: the
    lossless guide's resonance, left in place and named. 100 m rigid guide,
    c = 1500: cutoffs at (m - 1/2)·7.5 Hz = 41.25, 48.75, 56.25 … Hz."""

    @staticmethod
    def _run(frequencies):
        from uacpy import Environment, Receiver, Source, analytic
        from uacpy.core.boundary import BoundaryProperties
        env = Environment(bathymetry=100.0, ssp=1500.0, water_density=1.0,
                          bottom=BoundaryProperties(acoustic_type='rigid'))
        with recorded_warnings() as record:
            field = analytic.ideal_waveguide(
                env, Source(depths=30.0, frequencies=frequencies),
                Receiver(depths=[50.0], ranges=[3000.0]))
        return field, [str(w.message) for w in record]

    def test_a_grid_on_the_cutoffs_is_named(self):
        _, messages = self._run(np.arange(40.0, 60.0, 1.25))
        assert len(messages) == 1
        assert '41.25, 48.75, 56.25 Hz' in messages[0]

    def test_a_frequency_off_the_cutoffs_is_silent(self):
        field, messages = self._run([50.0, 56.26])
        assert messages == []
        assert np.isfinite(field.tl).all()

    @pytest.mark.parametrize('offset_hz, warns', [(1e-5, True),
                                                   (1e-4, False)])
    def test_the_warning_sits_at_k_r_times_r_max_of_one(self, offset_hz,
                                                        warns):
        """Above the 56.25 Hz cutoff at 3 km: +1e-5 Hz puts the eighth
        mode at |k_r|·r = 0.42 (warned), +1e-4 Hz at 1.33 (silent)."""
        f = 56.25 + offset_hz
        k_z = 7.5 * np.pi / 100.0
        k_r_r = np.sqrt((2 * np.pi * f / 1500.0) ** 2 - k_z ** 2) * 3000.0
        assert (k_r_r < 1.0) == warns and 0.3 < k_r_r < 2.0
        _, messages = self._run([f])
        assert len(messages) == (1 if warns else 0)


class TestEveryIpts68FitReceivesAnIpts68Temperature:
    """UNESCO, Del Grosso and EOS-80 were fitted on IPTS-68; each function
    takes ITS-90 and converts with Saunders' t68 = 1.00024*t90. At S = 0 and
    P = 0 only the temperature polynomial survives, so the function must
    equal that polynomial evaluated on t68 (the unconverted t90 is 0.012 m/s
    and 0.002 kg/m3 away at 20 C, far outside the tolerance)."""

    t90 = np.array([2.0, 10.0, 20.0, 30.0])

    def test_del_grosso_evaluates_its_polynomial_on_t68(self):
        t = self.t90 * 1.00024
        want = (1402.392 + 0.501109398873e1 * t - 0.550946843172e-1 * t ** 2
                + 0.221535969240e-3 * t ** 3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      # S = 0 is below the fit
            got = sound_speed_delgrosso(self.t90, 0.0, pressure_dbar=0.0)
        np.testing.assert_allclose(got, want, rtol=1e-14)

    def test_eos80_density_evaluates_its_pure_water_term_on_t68(self):
        from uacpy.core.acoustics import density
        t = self.t90 * 1.00024
        want = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
                + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4
                + 6.536332e-9 * t ** 5)
        np.testing.assert_allclose(density(self.t90, 0.0), want, rtol=1e-14)


class TestEveryLevelReadsAComplexRecordAsItIsNamed:
    """A complex record is an analytic signal or a baseband envelope, and
    the two give different levels, so ``spl``, ``peak_level`` and
    ``sound_exposure_level`` refuse one unless ``complex_record=`` names it.
    'analytic' reads the real part (the cosine's levels, not the envelope's
    +3.01 dB); 'baseband' reads |x| (mean square |x|²/2, peak |x|), which is
    independent of the carrier phase — the real part of a constant envelope
    is 116.99, 120.00 or -180 dB depending on that phase."""

    fs = 48_000.0
    t = np.arange(48_000) / 48_000.0
    x = np.cos(2 * np.pi * 1000.0 * t)
    z = x + 1j * np.sin(2 * np.pi * 1000.0 * t)

    def test_an_unnamed_complex_record_is_refused_by_every_level(self):
        from uacpy.core.acoustics import peak_level, sound_exposure_level, spl
        for call in (lambda: spl(self.z), lambda: peak_level(self.z),
                     lambda: sound_exposure_level(self.z, self.fs)):
            with pytest.raises(ConfigurationError, match='complex_record'):
                call()

    def test_the_analytic_record_gives_the_real_records_levels(self):
        from uacpy.core.acoustics import peak_level, sound_exposure_level, spl
        z = self.z * np.exp(1j * 0.3)       # real peak below the envelope
        assert spl(z, complex_record='analytic') == pytest.approx(
            spl(z.real), abs=1e-12)
        assert peak_level(z, complex_record='analytic') == pytest.approx(
            peak_level(z.real), abs=1e-12)
        assert sound_exposure_level(self.z, self.fs,
                                    complex_record='analytic') == \
            pytest.approx(sound_exposure_level(self.x, self.fs), abs=1e-12)

    @pytest.mark.parametrize('phase', [0.0, np.pi / 2, 1.0])
    def test_a_baseband_envelope_gives_the_passband_levels_at_any_phase(
            self, phase):
        from uacpy.core.acoustics import peak_level, sound_exposure_level, spl
        envelope = np.full(self.t.size, np.exp(1j * phase))  # cos at carrier
        assert spl(envelope, complex_record='baseband') == pytest.approx(
            spl(self.x), abs=1e-9)
        assert peak_level(envelope, complex_record='baseband') == \
            pytest.approx(peak_level(self.x), abs=1e-9)
        assert sound_exposure_level(envelope, self.fs,
                                    complex_record='baseband') == \
            pytest.approx(sound_exposure_level(self.x, self.fs), abs=1e-9)

    def test_a_real_record_needs_no_reading(self):
        from uacpy.core.acoustics import spl
        assert spl(self.x) == pytest.approx(
            20 * np.log10(np.sqrt(0.5) / 1e-6), abs=1e-9)


class TestEveryLevelFloorIsOnTheSquaredQuantity:
    """``floor`` bounds the squared pressure on ``spl``, ``peak_level`` and
    ``power_to_dB`` alike; ``transmission_loss_dB``'s bounds ``|p|`` and says
    so."""

    def test_one_floor_value_gives_one_level_on_the_three_level_helpers(self):
        from uacpy.core.acoustics import peak_level, power_to_dB, spl
        zeros = np.zeros(8)
        levels = [spl(zeros, floor=1e-20), peak_level(zeros, floor=1e-20),
                  float(power_to_dB(0.0, floor=1e-20))]
        np.testing.assert_allclose(levels, -80.0, atol=1e-9)

    def test_the_loss_floor_is_on_the_pressure_itself(self):
        from uacpy.core.acoustics import transmission_loss_dB
        assert float(transmission_loss_dB(0.0, floor=1e-10)) == pytest.approx(
            200.0)
        assert 'ITSELF' in transmission_loss_dB.__doc__


class TestTheHankelTransformAnnouncesTheAliasPeriod:
    """``hankel_transform`` is periodic in range with ``2*pi/dk``. Inside it
    the transform is quiet; from it out to fieldsco's ``10/dk`` it runs but
    warns, since folded energy cannot be told from real energy afterwards;
    past ``10/dk`` it refuses. dk = 0.01: alias period 628.3 m, limit 1000 m."""

    k = 0.01 * np.arange(1, 400)
    G = np.ones((1, 399), dtype=complex)

    def _warns(self, r_max):
        from uacpy.core.acoustics import hankel_transform
        with recorded_warnings() as caught:
            hankel_transform(self.G, self.k, np.array([100.0, r_max]),
                             attenuation=0.0)
        return any('alias period' in str(w.message) for w in caught)

    def test_inside_the_alias_period_is_quiet(self):
        assert not self._warns(620.0)

    def test_past_the_alias_period_warns(self):
        assert self._warns(640.0)
        assert self._warns(1000.0)

    def test_past_ten_over_dk_is_refused(self):
        from uacpy.core.acoustics import hankel_transform
        with pytest.raises(ConfigurationError, match='alias periods'):
            hankel_transform(self.G, self.k, np.array([1001.0]), attenuation=0.0)

    def test_the_public_name_is_its_own_function(self):
        import inspect
        from uacpy.core.acoustics import hankel_transform
        assert hankel_transform.__name__ == 'hankel_transform'
        assert 'who' not in inspect.signature(hankel_transform).parameters


def test_doppler_and_the_comms_scale_factor_are_one_relation():
    """``uacpy.acoustics.doppler`` and ``comms.doppler_from_speed`` share the default
    sound speed, so ``doppler(v, f) == f·(1 + doppler_from_speed(v))``."""
    from uacpy.comms import doppler_from_speed
    from uacpy.core.acoustics import doppler
    assert doppler(5.0, 10_000.0) == pytest.approx(
        10_000.0 * (1 + doppler_from_speed(5.0)), rel=1e-15)
    # The sound speed is the package's keyword, not ``c``.
    assert doppler(5.0, 10_000.0, sound_speed=1450.0) == pytest.approx(
        10_000.0 * (1 + 5.0 / 1450.0), rel=1e-15)
    assert 'doppler_from_speed' in doppler.__doc__


class TestSumLevelsIsThePowerSum:
    """``sum_levels_dB`` is ``10·log10(Σ 10^(L/10))`` evaluated without
    overflow: two equal levels add 10·log10(2), a -inf term adds nothing,
    and 4000 dB levels (past float64's 10**(L/10)) still sum."""

    def test_two_equal_levels_add_ten_log_two(self):
        from uacpy.core.acoustics import sum_levels_dB
        assert sum_levels_dB(60.0, 60.0) == pytest.approx(
            60.0 + 10 * np.log10(2.0), abs=1e-12)

    def test_it_matches_the_linear_sum_where_that_is_finite(self):
        from uacpy.core.acoustics import sum_levels_dB
        a, b = np.array([40.0, 70.0]), np.array([55.0, 20.0])
        np.testing.assert_allclose(
            sum_levels_dB(a, b, 30.0),
            10 * np.log10(10 ** (a / 10) + 10 ** (b / 10) + 10 ** 3.0),
            rtol=0, atol=1e-10)

    def test_minus_inf_contributes_nothing_and_loud_levels_do_not_overflow(self):
        from uacpy.core.acoustics import sum_levels_dB
        assert sum_levels_dB(50.0, -np.inf) == pytest.approx(50.0, abs=1e-12)
        assert sum_levels_dB(4000.0, 4000.0) == pytest.approx(
            4000.0 + 10 * np.log10(2.0))

    def test_no_level_is_refused(self):
        from uacpy.core.acoustics import sum_levels_dB
        with pytest.raises(ConfigurationError,
                           match='need at least one level'):
            sum_levels_dB()


class TestAPlainListIsAnArray:
    """A list is the array a user types most; every seawater and level
    helper takes one and gives what the same ndarray gives, as
    ``sound_speed_unesco`` always did."""

    def test_mackenzie_and_its_formula_route(self):
        from uacpy.core.acoustics import (sound_speed_at_depth,
                                          sound_speed_mackenzie)
        np.testing.assert_array_equal(
            sound_speed_mackenzie([10, 12], 35, [0, 100]),
            sound_speed_mackenzie(np.array([10.0, 12.0]), 35,
                                  np.array([0.0, 100.0])))
        np.testing.assert_array_equal(
            sound_speed_at_depth([10, 12], [35, 35], [0, 100],
                                 formula='mackenzie'),
            sound_speed_mackenzie(np.array([10.0, 12.0]), 35,
                                  np.array([0.0, 100.0])))

    def test_density(self):
        from uacpy.core.acoustics import density
        np.testing.assert_array_equal(density(10, [35, 36]),
                                      density(10, np.array([35.0, 36.0])))

    def test_doppler(self):
        from uacpy.core.acoustics import doppler
        np.testing.assert_array_equal(doppler([1, 2], 1000),
                                      doppler(np.array([1.0, 2.0]), 1000))

    def test_pressure(self):
        from uacpy.core.acoustics import pressure
        np.testing.assert_array_equal(
            pressure([100, 120], -170, 20),
            pressure(np.array([100.0, 120.0]), -170, 20))

    def test_a_scalar_keeps_the_scalar_path(self):
        from uacpy.core.acoustics import sound_speed_mackenzie
        assert type(sound_speed_mackenzie(10.0, 35.0, 0.0)) is float


class TestReceivedLevelKeepsTheNoEnergyMarker:
    """``received_level_dB`` is ``SL - TL``, except that a no-energy cell
    stays the marker: ``SL - 600`` is a finite level ``no_energy_mask``
    does not recognise, and a metric then averages it in."""

    def test_a_level_is_the_source_level_minus_the_loss(self):
        from uacpy.core.acoustics import received_level_dB
        assert received_level_dB(180.0, 60.0) == 120.0
        np.testing.assert_array_equal(
            received_level_dB(180.0, np.array([[40.0, 70.5]])),
            [[140.0, 109.5]])

    def test_a_no_energy_cell_is_the_level_view_of_the_marker(self):
        from uacpy.core.acoustics import no_energy_mask, received_level_dB
        from uacpy.core.constants import NO_ENERGY_DB
        tl = np.array([60.0, NO_ENERGY_DB, 325.0])
        level = received_level_dB(180.0, tl)
        np.testing.assert_array_equal(level, [120.0, -NO_ENERGY_DB, -145.0])
        np.testing.assert_array_equal(no_energy_mask(level),
                                      [False, True, False])


def test_the_wavenumber_helpers_use_the_package_argument_names():
    """``wavenumber_taper(k, frequency, …)`` and
    ``ranges_fit_alias_period(delta_k, rmax_m)``: the spellings every other
    public function uses for a frequency and a range extent."""
    import inspect
    from uacpy.core.acoustics import ranges_fit_alias_period, wavenumber_taper
    taper = inspect.signature(wavenumber_taper).parameters
    fit = inspect.signature(ranges_fit_alias_period).parameters
    assert 'frequency' in taper and 'freq' not in taper
    assert 'rmax_m' in fit and 'r_max' not in fit
    k = np.linspace(0.3, 0.5, 64)
    np.testing.assert_array_equal(
        wavenumber_taper(k, frequency=100.0, cmin=1450.0, cmax=1650.0),
        wavenumber_taper(k, 100.0, 1450.0, 1650.0))
    assert ranges_fit_alias_period(0.01, rmax_m=600.0)


def test_adiabatic_gradient_matches_the_unesco_check_value():
    """UNESCO 44 (Fofonoff & Millard 1983) publishes ATG(40, 40, 10000)."""
    from uacpy.core.acoustics.seawater import _adiabatic_gradient
    assert float(_adiabatic_gradient(40.0, 40.0, 10000.0)) == pytest.approx(
        3.255976e-4, rel=1e-6)


def test_potential_temperature_matches_the_unesco_check_value():
    """UNESCO 44 publishes THETA(S=40, T=40, P=10000, Pr=0) = 36.89073 degC."""
    from uacpy.core.acoustics.seawater import _shift_adiabatically
    assert float(_shift_adiabatically(40.0, 40.0, 10000.0, 0.0)) == pytest.approx(
        36.89073, abs=1e-5)


def test_insitu_from_potential_round_trips_over_the_ocean_range():
    from uacpy.core.acoustics.seawater import (_shift_adiabatically,
                                               insitu_from_potential)
    sal = np.array([33.0, 34.7, 35.5, 37.0])
    for theta in (-1.5, 1.2, 2.5, 10.0, 30.0):
        for pres in (500.0, 2000.0, 5000.0, 11000.0):
            insitu = insitu_from_potential(salinity=sal, theta=theta,
                                           pressure_dbar=pres)
            back = _shift_adiabatically(sal, insitu, pres, 0.0)
            assert np.allclose(back, theta, atol=2e-4)


def test_insitu_from_potential_takes_keywords_only():
    """A temperature-first positional call, (2.0, 35, 4000), read 2 °C as
    the salinity and returned 36.1 °C; by keyword the order cannot slip."""
    from uacpy.core.acoustics import insitu_from_potential
    got = insitu_from_potential(theta=2.0, salinity=35.0, pressure_dbar=4000.0)
    assert float(got) == pytest.approx(2.34, abs=0.01)
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        insitu_from_potential(2.0, 35.0, 4000.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'pres'"):
        insitu_from_potential(salinity=34.7, theta=1.5, pres=5000.0)


def test_insitu_from_potential_is_warmer_and_grows_with_pressure():
    """Compression warms a parcel, so in-situ exceeds potential below 0 dbar."""
    from uacpy.core.acoustics.seawater import insitu_from_potential
    pres = np.array([0.0, 1000.0, 5000.0, 10000.0])
    excess = insitu_from_potential(salinity=34.7, theta=1.5,
                                   pressure_dbar=pres) - 1.5
    assert excess[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(excess) > 0.0)
    assert excess[2] == pytest.approx(0.450, abs=0.01)


def test_potential_temperature_costs_about_two_m_per_s_at_5000_dbar():
    """Potential temperature in place of in-situ temperature costs about
    2 m/s of sound speed at 5000 dbar, in every pressure equation."""
    from uacpy.core.acoustics.seawater import (SOUND_SPEED_FORMULAS,
                                               insitu_from_potential)
    sal, theta, pres = 34.7, 1.5, 5000.0
    insitu = float(insitu_from_potential(salinity=sal, theta=theta,
                                         pressure_dbar=pres))
    for name in ('unesco', 'delgrosso', 'teos10'):
        speed_fn = SOUND_SPEED_FORMULAS[name]
        delta = speed_fn(insitu, sal, pres) - speed_fn(theta, sal, pres)
        assert 1.8 < delta < 2.0, (name, delta)


# ── integrate_psd / band_level, and the marker through sum_levels_dB ──────


def test_integrate_psd_splices_the_band_edges_into_the_grid():
    """A flat density's band power is density × width exactly, whatever the
    grid, because the edges are nodes of the trapezoid."""
    from uacpy.core.acoustics import band_level, integrate_psd
    f = np.linspace(0.0, 1000.0, 11)
    assert integrate_psd(np.full(11, 2.0), f, 123.0, 456.0) == \
        pytest.approx(2.0 * 333.0, rel=1e-12)
    assert integrate_psd(np.full(11, 2.0), f) == pytest.approx(2000.0)
    assert band_level(np.full(11, 20.0), f, 100.0, 200.0) == \
        pytest.approx(40.0, abs=1e-9)
    assert band_level(np.full(11, -np.inf), f, 100.0, 200.0) == -np.inf


@pytest.mark.parametrize('band', [(-1.0, 100.0), (100.0, 1000.5),
                                  (500.0, 500.0)])
def test_integrate_psd_refuses_a_band_it_cannot_integrate_whole(band):
    from uacpy.core.acoustics import integrate_psd
    from uacpy.core.exceptions import ConfigurationError
    f = np.linspace(0.0, 1000.0, 11)
    with pytest.raises(ConfigurationError, match='integrate_psd'):
        integrate_psd(np.ones(11), f, *band)


def test_integrate_psd_takes_a_band_ending_on_the_grid_edge():
    from uacpy.core.acoustics import integrate_psd
    f = np.linspace(0.0, 1000.0, 11)
    assert integrate_psd(np.ones(11), f, 900.0, 1000.0) == pytest.approx(100.0)


def test_the_sum_of_no_energy_markers_is_the_marker():
    """Two -600 dB markers sum to -596.99 by the formula, which no_energy_mask
    reads as a level; the sum keeps the marker. A real level beside a marker
    is the real level, and past-the-marker sums stay what they are."""
    from uacpy.core.acoustics import no_energy_mask, sum_levels_dB
    from uacpy.core.constants import NO_ENERGY_DB
    assert sum_levels_dB(-NO_ENERGY_DB, -NO_ENERGY_DB) == -NO_ENERGY_DB
    assert no_energy_mask(sum_levels_dB(-NO_ENERGY_DB, -NO_ENERGY_DB))
    assert sum_levels_dB(60.0, -NO_ENERGY_DB) == pytest.approx(60.0)
    assert sum_levels_dB(-1200.0, -1200.0) == pytest.approx(-1196.9897,
                                                             abs=1e-3)
    np.testing.assert_allclose(
        sum_levels_dB(np.array([-NO_ENERGY_DB, 50.0]),
                      np.array([-NO_ENERGY_DB, 50.0])),
        [-NO_ENERGY_DB, 53.0103], atol=1e-4)


def test_a_nan_level_sums_to_nan_without_a_warning():
    import warnings
    from uacpy.core.acoustics import sum_levels_dB
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = sum_levels_dB(np.array([np.nan, 60.0]), np.array([10.0, 60.0]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(63.0103, abs=1e-4)


@pytest.mark.parametrize('call, expected', [
    (lambda f: f([100.0, 100.0, 100.0], axis=0), 104.771),
    (lambda f: f(100.0, 100.0, 100.0), 104.771),
    (lambda f: f(np.array([[60.0, 70.0], [60.0, 70.0]]), axis=1),
     [70.414, 70.414]),
    (lambda f: f([100.0, 100.0, 100.0]), [100.0, 100.0, 100.0])])
def test_sum_levels_dB_sums_one_array_only_along_a_named_axis(call, expected):
    """``sum_levels_dB([L1, L2, L3])`` returned the list unchanged: one array
    is one level. ``axis=`` sums its entries; varargs still do."""
    from uacpy.core.acoustics import sum_levels_dB
    np.testing.assert_allclose(call(sum_levels_dB), expected, atol=1e-3)


def test_sum_levels_dB_takes_axis_with_one_argument_only():
    from uacpy.core.acoustics import sum_levels_dB
    with pytest.raises(ConfigurationError, match='got 2 arguments'):
        sum_levels_dB([1.0, 2.0], [3.0, 4.0], axis=0)


# ── the one water-property rule ───────────────────────────────────────────

#: (depth or pressure, value) pairs and the evaluation coordinate.
_T_PAIRS = [(0.0, 22.0), (60.0, 12.0), (200.0, 10.0)]
_S_PAIRS = [(0.0, 35.5), (300.0, 34.6)]
_AT = np.array([0.0, 30.0, 100.0, 500.0])
_T_AT = np.interp(_AT, *np.array(_T_PAIRS).T)
_S_AT = np.interp(_AT, *np.array(_S_PAIRS).T)


def _entry_points():
    """Every formula entry point with water-property inputs, as
    ``(name, call(T, S, at))``: the coordinate is depth, or pressure for an
    equation stated in pressure."""
    from uacpy.core.acoustics import attenuation as att, seawater as sw
    return {
        'sound_speed_mackenzie': lambda t, s, at: sw.sound_speed_mackenzie(
            t, s, depth=at),
        'sound_speed_unesco': lambda t, s, at: sw.sound_speed_unesco(
            t, s, depth=at),
        'sound_speed_delgrosso': lambda t, s, at: sw.sound_speed_delgrosso(
            t, s, depth=at),
        'sound_speed_teos10': lambda t, s, at: sw.sound_speed_teos10(
            t, s, depth=at),
        'sound_speed_at_depth': lambda t, s, at: sw.sound_speed_at_depth(
            t, s, at),
        'absorption_francois_garrison':
            lambda t, s, at: att.absorption_francois_garrison(
                1e4, t, s, 8.0, depth=at),
    }


@pytest.mark.parametrize('name', sorted(_entry_points()))
def test_pairs_are_the_values_interpolated_onto_the_coordinate(name):
    """Pairs, arrays, single values and a mix give the same answer as the
    same water given as arrays on the coordinate."""
    call = _entry_points()[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want = np.asarray(call(_T_AT, _S_AT, _AT))
        np.testing.assert_array_equal(call(_T_PAIRS, _S_PAIRS, _AT), want)
        np.testing.assert_array_equal(call(_T_PAIRS, _S_AT, _AT), want)
        one = np.asarray(call(_T_PAIRS, 35.0, _AT))
        np.testing.assert_array_equal(
            one, np.asarray(call(_T_AT, np.full(4, 35.0), _AT)))
        # A single value and an array broadcast as they always did.
        assert np.ndim(call(10.0, 35.0, 100.0)) == 0 or \
            np.size(call(10.0, 35.0, 100.0)) == 1


# sound_speed_at_depth takes depth as a required argument, so there is no
# call without the coordinate to refuse; the other entry points are checked.
@pytest.mark.parametrize(
    'name', sorted(set(_entry_points()) - {'sound_speed_at_depth'}))
def test_pairs_without_the_coordinate_are_refused(name):
    from uacpy.core.acoustics import attenuation as att, seawater as sw
    calls = {
        'sound_speed_mackenzie': lambda: sw.sound_speed_mackenzie(_T_PAIRS),
        'sound_speed_unesco': lambda: sw.sound_speed_unesco(_T_PAIRS),
        'sound_speed_delgrosso': lambda: sw.sound_speed_delgrosso(_T_PAIRS),
        'sound_speed_teos10': lambda: sw.sound_speed_teos10(_T_PAIRS),
        'absorption_francois_garrison':
            lambda: att.absorption_francois_garrison(1e4, _T_PAIRS),
    }
    with pytest.raises(ConfigurationError, match=r'pairs.*pass (depth|pressure)='):
        calls[name]()


def test_a_grid_shaped_like_its_coordinate_is_not_pairs():
    """An (N, 2) value the same shape as the coordinate is a grid of
    values, broadcast as before."""
    from uacpy.core.acoustics import sound_speed_mackenzie
    t = np.array([[10.0, 12.0], [8.0, 9.0]])
    z = np.array([[0.0, 0.0], [100.0, 100.0]])
    np.testing.assert_array_equal(
        sound_speed_mackenzie(t, 35.0, z),
        [[sound_speed_mackenzie(10.0, 35.0, 0.0),
          sound_speed_mackenzie(12.0, 35.0, 0.0)],
         [sound_speed_mackenzie(8.0, 35.0, 100.0),
          sound_speed_mackenzie(9.0, 35.0, 100.0)]])


def test_every_water_entry_point_uses_the_shared_helper():
    """Structural: each entry point routes each water property through
    ``water_property`` — directly, or through ``_water_at_pressure`` (which
    does) or the law's ``_local`` (which does) — so none can drift to a rule
    of its own."""
    import ast
    import inspect
    import textwrap
    from uacpy.core import absorption, ssp
    from uacpy.core.acoustics import attenuation as att, seawater as sw

    def routed(func):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = (node.func.attr if isinstance(node.func, ast.Attribute)
                      else getattr(node.func, 'id', ''))
            args = [a for a in node.args]
            if callee == 'water_property' and args:
                names.add(ast.unparse(args[0]))
            elif callee == '_water_at_pressure':
                names.update(ast.unparse(a) for a in args[:2])
            elif callee == '_local' and args:
                names.add(ast.unparse(args[0]))
        return names

    for helper in (sw._water_at_pressure, absorption.FrancoisGarrison._local):
        assert routed(helper), helper.__qualname__
    expect = {
        sw.sound_speed_mackenzie: {'temperature', 'salinity'},
        sw.sound_speed_unesco: {'temperature', 'salinity'},
        sw.sound_speed_delgrosso: {'temperature', 'salinity'},
        sw.sound_speed_teos10: {'temperature', 'salinity'},
        sw.sound_speed_at_depth: {'temperature', 'salinity'},
        att.absorption_francois_garrison: {'temperature', 'salinity', 'pH'},
        absorption.FrancoisGarrison._alpha_dB_per_m:
            {"'temperature'", "'salinity'"},
        absorption.FrancoisGarrison._nbs_at:
            {"'pH'", "'temperature'", "'salinity'"},
        ssp.SoundSpeedProfile.from_temperature_salinity: {'value'},
    }
    for func, params in expect.items():
        assert params <= routed(func), (func.__qualname__, routed(func))
    # Every public sound_speed_* formula taking T and S is in the list.
    public = {n for n in sw.__all__ if n.startswith('sound_speed_')}
    listed = {f.__name__ for f in expect if f.__name__.startswith('sound_')}
    assert public == listed


def test_thorp_takes_the_francois_garrison_call_shape():
    """``absorption_thorp(f, depth=)`` broadcasts like
    ``absorption_francois_garrison(f, depth=)``; with no depth term its
    value is the same at every depth, and without ``depth`` it keeps the
    shape of ``f``."""
    from uacpy.core.acoustics import (
        absorption_francois_garrison, absorption_thorp)
    f = np.array([1e3, 1e4])[:, None]
    z = np.array([0.0, 100.0, 4000.0])
    got = absorption_thorp(f, depth=z)
    assert got.shape == np.shape(absorption_francois_garrison(f, depth=z))
    np.testing.assert_array_equal(got, np.repeat(absorption_thorp(f), 3,
                                                 axis=1))
    assert np.shape(absorption_thorp(1e3)) == ()


@pytest.mark.parametrize('name', ['sound_speed_unesco', 'sound_speed_delgrosso',
                                  'sound_speed_teos10'])
def test_a_pressure_equation_takes_depth_pairs_and_converts_them(name):
    """The equations stated in pressure take ``depth=`` like every other
    formula, with ``(depth, value)`` pairs on it: the answer is the equation
    at the pressure ``depth_to_pressure_dbar`` gives, with the water
    interpolated by hand, at the 45° default and at a latitude given."""
    from uacpy.core.acoustics import seawater as sw
    eq = getattr(sw, name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for lat in (None, 70.0):
            p = sw.depth_to_pressure_dbar(_AT, 45.0 if lat is None else lat)
            want = eq(_T_AT, _S_AT, pressure_dbar=p)
            got = eq(_T_PAIRS, _S_PAIRS, depth=_AT, latitude_deg=lat)
            np.testing.assert_array_equal(got, want)
        # Below the surface the latitude moves the pressure, so the speed.
        assert np.all(eq(_T_AT, _S_AT, depth=_AT, latitude_deg=70.0)[1:]
                      != eq(_T_AT, _S_AT, depth=_AT)[1:])


@pytest.mark.parametrize('name', ['sound_speed_unesco', 'sound_speed_delgrosso',
                                  'sound_speed_teos10'])
def test_a_pressure_equation_refuses_ambiguous_coordinates(name):
    from uacpy.core.acoustics import seawater as sw
    eq = getattr(sw, name)
    with pytest.raises(ConfigurationError, match='not both'):
        eq(10.0, 35.0, depth=100.0, pressure_dbar=100.0)
    with pytest.raises(ConfigurationError, match='no depth= was given'):
        eq(10.0, 35.0, pressure_dbar=100.0, latitude_deg=45.0)
    with pytest.raises(ConfigurationError, match='pass depth='):
        eq(_T_PAIRS, 35.0, pressure_dbar=100.0)
    with pytest.raises(TypeError, match='positional'):  # keyword-only
        eq(10.0, 35.0, 100.0)


def test_sound_speed_at_depth_requires_its_depth():
    """Its depth is a required argument, so a call without it cannot reach
    the pairs rule: Python refuses it first."""
    from uacpy.core.acoustics import seawater as sw
    import inspect
    depth = inspect.signature(sw.sound_speed_at_depth).parameters['depth']
    assert depth.default is inspect.Parameter.empty
