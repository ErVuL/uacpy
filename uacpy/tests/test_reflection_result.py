"""``ReflectionCoefficient``, the tabulated reflection result.

Magnitude and phase, the travelling-wave phase sign, transmission values
above one, label slicing, and how ``eval`` interpolates and extrapolates.
"""

import numpy as np
import pytest
import warnings
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import ReflectionCoefficient


class TestReflectionCoefficientCarriesTransmissionValuesAboveOne:
    """``R`` is in [0, 1] for a reflection coefficient, but OASR's
    ``reflection_type='transmission'`` column is a *transmission* coefficient
    — an amplitude ratio across the interface, which exceeds 1 into a
    higher-impedance medium (Medwin & Clay: air→water gives T12 ≈ 2). The
    class neither clamps nor warns, so those samples reach the caller intact,
    and a clamp added later would silently corrupt every transmission run.
    """

    ABOVE_ONE = 1.15   # the peak PV measured on a 1700/400 m/s half-space

    def _rc(self):
        from uacpy.core.results import ReflectionCoefficient
        theta = np.linspace(0.0, 90.0, 5)
        R = np.array([0.3, 0.8, self.ABOVE_ONE, 0.9, 0.0])
        return ReflectionCoefficient(
            angles=theta, magnitude=R, phase=np.zeros_like(R),
            model='OASR', frequencies=100.0,
            reflection_type='transmission',
        )

    def test_a_sample_above_one_survives_construction(self):
        assert float(self._rc().magnitude.max()) == pytest.approx(self.ABOVE_ONE)

    def test_a_sample_above_one_survives_slicing(self):
        sliced = self._rc().at(angle=45.0)
        assert float(sliced.magnitude[0]) == pytest.approx(self.ABOVE_ONE)

    def test_a_reflection_run_is_not_given_a_bound_it_never_had(self):
        """Both sides: the class applies no bound either way, so a physical
        reflection table below 1 passes through unchanged too."""
        from uacpy.core.results import ReflectionCoefficient
        R = np.array([0.3, 0.8, 0.95, 0.9, 0.0])
        rc = ReflectionCoefficient(
            angles=np.linspace(0.0, 90.0, 5), magnitude=R, phase=np.zeros_like(R),
            model='OASR', frequencies=100.0,
            reflection_type='P-P',
        )
        np.testing.assert_allclose(rc.magnitude, R)


class TestReflectionCoefficientChainAccessors:
    """``ReflectionCoefficient.at`` — label slicing of the angle and
    frequency axes; broadband-only kwargs raise on narrowband instances."""

    def _broadband_rc(self):
        from uacpy.core.results import ReflectionCoefficient
        theta = np.linspace(0, 90, 91)            # 91 angles
        freqs = np.array([50.0, 100.0, 200.0])    # 3 frequencies
        R = np.outer(np.cos(np.deg2rad(theta)), np.ones(3)) ** 2
        phi = np.zeros_like(R)
        return ReflectionCoefficient(
            angles=theta, magnitude=R, phase=phi,
            frequencies=freqs, model='Test',
        )

    def test_at_frequency_returns_narrowband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(frequency=100.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert not sliced.is_broadband
        assert sliced.magnitude.shape == (91,)

    def test_at_angle_keeps_broadband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert sliced.angles.shape == (1,)
        assert sliced.magnitude.shape == (1, 3)

    def test_at_both_collapses_to_single_value(self):
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0, frequency=100.0)
        assert sliced.angles.shape == (1,)
        assert sliced.magnitude.shape == (1,)

    def test_at_frequency_on_narrowband_raises(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = ReflectionCoefficient(
            angles=np.linspace(0, 90, 5),
            magnitude=np.linspace(0, 1, 5),
            phase=np.zeros(5),
            model='Test', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError, match="broadband"):
            rc.at(frequency=100.0)

    @pytest.mark.parametrize('slice_call', [
        lambda rc: rc.at(frequency=100.0),
        lambda rc: rc.at(angle=45.0),
        lambda rc: rc.isel(frequency=1),
        lambda rc: rc.isel(angle=10),
        lambda rc: rc.eval(frequency=120.0),
    ])
    def test_slicing_keeps_provenance(self, slice_call):
        """Every slice carries ``model_source`` (the plot credit) and
        ``phase_reference`` — the same invariant ``Field.id_kwargs``
        enforces."""
        from uacpy.models.provenance import model_provenance
        src = model_provenance('acoustics_toolbox')
        rc = self._broadband_rc()
        rc.model_source = src
        rc.phase_reference = 'travelling_wave'
        sliced = slice_call(rc)
        assert sliced.model_source is src
        assert sliced.phase_reference == 'travelling_wave'

    def test_at_angle_selects_one_angle(self):
        rc = self._broadband_rc()
        by_angle = rc.at(angle=45.0)
        assert by_angle.angles.shape == (1,)

    def test_at_unknown_axis_raises(self):
        # Generic Field.at-style form rejects axes it doesn't have.
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="unknown axis"):
            rc.at(depth=5.0)

    def test_theta_is_not_an_axis_name(self):
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="unknown axis"):
            rc.at(**{'theta': 45.0})

    def test_isel_positional_angle(self):
        rc = self._broadband_rc()
        s = rc.isel(angle=2)
        assert s.angles.shape == (1,) and s.angles[0] == rc.angles[2]
        assert s.magnitude.shape == (1, 3)

    def test_isel_oob_raises_indexerror(self):
        with pytest.raises(IndexError, match='is out of bounds for axis 0'):
            self._broadband_rc().isel(angle=999)

    def test_eval_interpolates_off_grid_angle(self):
        rc = self._broadband_rc()              # 1° grid → 30.5° is off-grid
        s = rc.eval(angle=30.5)
        assert s.angles[0] == pytest.approx(30.5)
        expect = 0.5 * (np.cos(np.deg2rad(30)) ** 2
                        + np.cos(np.deg2rad(31)) ** 2)   # linear of cos^2
        assert s.magnitude[0, 0] == pytest.approx(expect, abs=1e-6)

    def test_eval_method_cubic(self):
        rc = self._broadband_rc()
        s = rc.eval(angle=30.5, method='cubic')
        assert s.magnitude[0, 0] == pytest.approx(np.cos(np.deg2rad(30.5)) ** 2, abs=1e-3)


class TestReflectionCoefficientExtrapolationIsAnnounced:
    """``eval``/``at`` hold the end value outside the tabulated angles, like
    every other carrier. A solver reading the same table does not:
    ``misc/RefCoef.f90:139-140,146-147`` sets R = 0 and phi = 0 outside the
    range, killing the ray. The number is deliberately unchanged — returning 0
    would make this the only carrier with a different extrapolation rule, and 0
    is AT's kill convention, not a claim about R at that angle — but the
    disagreement is named in the warning."""

    @staticmethod
    def _rc():
        th = np.arange(10.0, 41.0, 5.0)
        return ReflectionCoefficient(
            angles=th, magnitude=np.linspace(0.9, 0.3, th.size),
            phase=np.zeros(th.size), model='Test', frequencies=100.0)

    @pytest.mark.parametrize('angle', [5.0, 80.0])
    def test_out_of_range_angle_warns_and_cites_the_solver(self, angle):
        with pytest.warns(UserWarning, match='RefCoef.f90'):
            self._rc().eval(angle=angle)

    def test_in_range_angle_is_silent_and_interpolates(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = self._rc().eval(angle=25.0)
        assert float(np.asarray(out.magnitude).ravel()[0]) == pytest.approx(0.6)

    def test_isel_takes_an_index_and_must_not_warn(self):
        # The discriminating case: isel's `angle` is a positional index, so
        # comparing it against degrees is meaningless. A guard on the shared
        # funnel that forgets this fires a bogus warning on every isel.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._rc().isel(angle=2)


class TestReflectionEvalInterpolatesPhaseAsAtDoesAndSaysSo:
    """``eval`` blends the phase column linearly, which is what
    ``misc/RefCoef.f90:167`` does, and therefore inherits the precondition AT
    states at ``misc/RefCoef.f90:119``: "Assumes phi has been unwrapped so
    that it varies smoothly". uacpy documented the arithmetic and not the
    assumption, so a table carrying the ±π branch cut interpolated through
    zero and reported a phase reversal as no phase shift.

    The behaviour is pinned as it is, not fixed: unwrapping here would put
    uacpy's answer at odds with every solver reading the same table. What is
    new is that :meth:`eval` now says so, and that the number in its docstring
    is the number this measures."""

    @staticmethod
    def _rc(phi):
        return ReflectionCoefficient(angles=np.array([10.0, 20.0, 30.0]),
                                     magnitude=np.array([0.9, 0.8, 0.7]),
                                     phase=np.asarray(phi, dtype=float))

    @staticmethod
    def _phi_at(rc, angle):
        return float(np.ravel(rc.eval(angle=angle).phase)[0])

    def test_a_wrapped_table_interpolates_through_zero_across_the_cut(self):
        # 3.0 and -3.0 are 0.283 rad apart the long way round; read as plain
        # reals their midpoint is 0.
        assert self._phi_at(self._rc([2.5, 3.0, -3.0]), 25.0) == 0.0

    def test_the_unwrapped_table_returns_the_physical_phase_instead(self):
        unwrapped = np.unwrap(np.array([2.5, 3.0, -3.0]))
        assert self._phi_at(self._rc(unwrapped), 25.0) == pytest.approx(
            np.pi, abs=1e-9)

    def test_the_two_agree_away_from_the_cut(self):
        # The discriminating half: the divergence is the branch cut, not
        # interpolation in general.
        wrapped, unwrapped = [2.5, 3.0, -3.0], np.unwrap([2.5, 3.0, -3.0])
        assert self._phi_at(self._rc(wrapped), 15.0) == pytest.approx(2.75)
        assert self._phi_at(self._rc(unwrapped), 15.0) == pytest.approx(2.75)

    def test_the_blend_is_the_formula_the_toolbox_uses(self):
        # RefCoef.f90:167, phi = (1-alpha)*phi_left + alpha*phi_right, with
        # alpha the fractional position between the bracketing abscissas.
        rc = self._rc([2.5, 3.0, -3.0])
        alpha = (12.5 - 10.0) / (20.0 - 10.0)
        assert self._phi_at(rc, 12.5) == pytest.approx(
            (1 - alpha) * 2.5 + alpha * 3.0)

    def test_nearest_and_at_are_untouched_by_the_cut(self):
        rc = self._rc([2.5, 3.0, -3.0])
        # Both return a tabulated sample, so neither can land between branches.
        assert float(np.ravel(rc.eval(angle=25.0, method='nearest').phase)[0]) \
            in (3.0, -3.0)
        assert float(np.ravel(rc.at(angle=25.0).phase)[0]) in (3.0, -3.0)

    def test_eval_documents_the_precondition_and_cites_the_toolbox_line(self):
        doc = ' '.join(ReflectionCoefficient.eval.__doc__.split())
        assert 'unwrapped' in doc
        assert 'misc/RefCoef.f90:119' in doc
        assert 'eval(angle=25)' in doc and '3.1416' in doc


class TestReflectionCoefficientCarriesTheTravellingWavePhaseSign:
    """``phi`` is written and read in the package's travelling-wave sign
    (positive below the critical angle on a lossy fluid half-space, as
    Bounce writes it and the AT engines read it), so a result built
    without an explicit convention is stamped ``'travelling_wave'``; an
    explicit stamp is kept."""

    @staticmethod
    def _rc(**kwargs):
        from uacpy.core.results import ReflectionCoefficient
        theta = np.array([10.0, 20.0, 30.0])
        return ReflectionCoefficient(angles=theta, magnitude=np.full(3, 0.5),
                                     phase=np.zeros(3), model='Test', **kwargs)

    def test_default_stamp_is_travelling_wave(self):
        assert self._rc().phase_reference == 'travelling_wave'

    def test_an_explicit_stamp_is_kept(self):
        rc = self._rc(phase_reference='time_domain_native')
        assert rc.phase_reference == 'time_domain_native'

    def test_a_slice_keeps_the_stamp(self):
        assert self._rc().at(angle=20.0).phase_reference == 'travelling_wave'


class TestAReflectionTableHasItsComplexAndLossViews:
    """``ReflectionCoefficient.coefficient`` is ``R·exp(iφ)`` and ``.dB`` is
    ``-20·log10 R`` (negative where a transmission table exceeds 1), so the
    two things a user wants from the table need no hand arithmetic."""

    @staticmethod
    def _table(R):
        from uacpy.core.results import ReflectionCoefficient
        return ReflectionCoefficient(angles=[10.0, 45.0, 80.0], magnitude=R,
                                     phase=[0.3, -0.2, 0.0], frequencies=[100.0])

    def test_the_complex_view_carries_magnitude_and_phase(self):
        t = self._table([0.9, 0.5, 0.1])
        np.testing.assert_allclose(np.abs(t.coefficient), t.magnitude)
        np.testing.assert_allclose(np.angle(t.coefficient), t.phase)

    def test_the_loss_view_is_minus_twenty_log_r(self):
        t = self._table([1.0, 0.5, 1.15])
        np.testing.assert_allclose(t.dB, -20 * np.log10([1.0, 0.5, 1.15]))
        assert t.dB[0] == 0.0 and t.dB[2] < 0.0


class TestAReflectionCoefficientIsAMagnitudeAndAPhase:
    """``R`` is the magnitude and ``phi`` the phase: a complex ``R`` kept its
    real part (not |R|) and lost the phase behind a bare ComplexWarning
    (RES-8); ``from_complex`` is the complex entry point (A1)."""

    def test_a_complex_r_is_refused_with_the_constructor_to_use(self):
        with pytest.raises(ConfigurationError, match='from_complex'):
            ReflectionCoefficient(angles=[10.0, 20.0],
                                  magnitude=[0.9 * np.exp(0.3j), 0.8 * np.exp(0.5j)],
                                  phase=[0.0, 0.0])

    def test_a_negative_magnitude_and_a_nan_angle_are_refused(self):
        with pytest.raises(ConfigurationError, match='negative'):
            ReflectionCoefficient(angles=[10.0, 20.0], magnitude=[-0.5, 0.5],
                                  phase=[0.0, 0.0])
        with pytest.raises(ConfigurationError, match='non-finite angle'):
            ReflectionCoefficient(angles=[10.0, np.nan], magnitude=[0.5, 0.5],
                                  phase=[0.0, 0.0])

    def test_from_complex_keeps_the_magnitude_and_unwraps_the_phase(self):
        theta = np.array([10.0, 20.0, 30.0])
        phase = np.array([2.5, 3.0, 3.5])       # crosses +pi
        R = np.array([0.9, 0.8, 0.7]) * np.exp(1j * phase)
        table = ReflectionCoefficient.from_complex(theta, R)
        np.testing.assert_allclose(table.magnitude, [0.9, 0.8, 0.7], rtol=1e-15)
        np.testing.assert_allclose(table.phase, phase, rtol=1e-15)
        np.testing.assert_allclose(table.coefficient, R, rtol=1e-15)

    def test_a_broadband_table_unwraps_along_the_angle_per_frequency(self):
        theta = np.array([10.0, 20.0, 30.0])
        phase = np.array([[2.5, -1.0], [3.0, -1.2], [3.5, -1.4]])
        R = 0.5 * np.exp(1j * phase)
        table = ReflectionCoefficient.from_complex(theta, R,
                                                   frequencies=[100.0, 200.0])
        np.testing.assert_allclose(table.phase, phase, rtol=1e-15)

    def test_isel_takes_an_integer_position(self):
        table = ReflectionCoefficient(angles=[10.0, 20.0], magnitude=[0.9, 0.8],
                                      phase=[0.3, 0.5])
        with pytest.raises(ConfigurationError, match='not an integer index'):
            table.isel(angle=1.5)
