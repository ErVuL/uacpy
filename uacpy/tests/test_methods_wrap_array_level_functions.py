"""Each carrier method that computes something generic calls a function a
user can call too, and the two agree.

The rule these pin: if a computation makes sense on data obtained anywhere
else — another model, a file, a measurement — it exists as an exported
function taking plain arrays, and the object's method is a wrapper around
it. Two failures the rule prevents, both seen in this package:

* **duplicated** — two implementations drift. ``Field.to_transfer_function``
  and ``transfer_function_from_impulse_response`` were written with
  different normalisations and disagreed by a factor of ``fs`` while the
  phase stayed exact to 1e-16, so every phase assertion passed.
* **missing** — a computation reachable only through a carrier is one a
  user holding the same numbers cannot do at all.

The delegation tests below read the method's source and assert the
arithmetic is not there. That is deliberately a structural check: a value
test passes on a copy that happens to agree today.
"""

import inspect

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    arrival_transfer_function,
    modal_group_velocity,
    broadband_propagation_loss,
    channel_regime,
    coherence_bandwidth,
    energy_support,
    gate_transfer_function,
    rms_delay_spread,
    tone_phasor,
)
from uacpy.comms import pulse_shaped_taps
from uacpy.core.acoustics import (
    modal_attenuation, modal_field, peak_level, power_to_dB,
    sound_exposure_level, spl,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results.field import Field
from uacpy.core.results.modes import Modes
from uacpy.core.results.rays import Arrivals
from uacpy.core.source import Source


def _arrivals(delays, amps, *, imag=None, phases=None):
    n = len(delays)
    cell = {"delays": np.asarray(delays, float),
            "delays_imag": np.zeros(n) if imag is None else np.asarray(imag, float),
            "amplitudes": np.asarray(amps, float),
            "phases": np.zeros(n) if phases is None else np.asarray(phases, float),
            "n_top_bounces": np.zeros(n, int), "n_bot_bounces": np.zeros(n, int),
            "src_angles": np.zeros(n), "rcv_angles": np.zeros(n)}
    return Arrivals(by_receiver=[[[cell]]], receiver_depths=np.array([50.0]),
                    receiver_ranges=np.array([1000.0]), model='Test',
                    frequencies=100.0)


def _body(method):
    """A method's source past its docstring."""
    src = inspect.getsource(method)
    return src.split('"""')[2] if src.count('"""') >= 2 else src


class TestThePowerDelayProfileStatisticsAreReachableWithoutArrivals:
    """Four textbook statistics that were defined only inside ``Arrivals``.
    A user with a measured power delay profile — from a chirp sounding, from
    a file — could not reach any of them."""

    DELAYS = np.array([0.010, 0.0135, 0.021, 0.0402])
    AMPS = np.array([1.0, 0.55, 0.31, 0.07])

    @property
    def POWERS(self):
        return self.AMPS ** 2

    def test_the_spread_is_the_weighted_second_central_moment(self):
        w = self.POWERS / self.POWERS.sum()
        mean = float((w * self.DELAYS).sum())
        expected = float(np.sqrt((w * (self.DELAYS - mean) ** 2).sum()))
        assert rms_delay_spread(self.DELAYS, self.POWERS) == pytest.approx(
            expected, rel=1e-15)

    def test_the_spread_is_invariant_to_the_power_scale(self):
        """It is a width, not a level: any consistent scaling of the powers
        must give the same answer, which is what lets a caller pass |a|**2
        in whatever units they have."""
        a = rms_delay_spread(self.DELAYS, self.POWERS)
        b = rms_delay_spread(self.DELAYS, self.POWERS * 1e7)
        assert b == pytest.approx(a, rel=1e-12)

    def test_the_span_holds_the_share_of_energy_asked_for(self):
        order = np.argsort(self.DELAYS)
        cum = np.cumsum(self.POWERS[order]) / self.POWERS.sum()
        cut = int(np.searchsorted(cum, 0.999, 'left'))
        expected = float(self.DELAYS[order][cut] - self.DELAYS[order][0])
        assert energy_support(self.DELAYS, self.POWERS) == pytest.approx(
            expected, rel=1e-15)

    def test_a_smaller_share_never_needs_a_longer_span(self):
        """The other side of the same threshold: the span is monotone in
        the fraction, so a test at one fraction alone cannot tell a correct
        search from one that returns the peak-to-peak span every time."""
        spans = [energy_support(self.DELAYS, self.POWERS, f)
                 for f in (0.5, 0.9, 0.999, 1.0)]
        assert spans == sorted(spans)
        assert spans[0] < spans[-1]
        assert spans[-1] == pytest.approx(float(np.ptp(self.DELAYS)))

    def test_the_coherence_bandwidth_is_the_reciprocal_of_the_spread(self):
        spread = rms_delay_spread(self.DELAYS, self.POWERS)
        assert coherence_bandwidth(self.DELAYS, self.POWERS) == pytest.approx(
            1.0 / spread, rel=1e-15)
        assert coherence_bandwidth(
            self.DELAYS, self.POWERS, convention='rappaport_0.9'
        ) == pytest.approx(1.0 / (50.0 * spread), rel=1e-15)

    def test_a_flat_channel_has_unbounded_coherence_bandwidth(self):
        assert coherence_bandwidth([0.01], [1.0]) == float('inf')

    @pytest.mark.parametrize('rate,selective', [(10.0, False), (5000.0, True)])
    def test_the_regime_compares_the_signal_band_with_the_coherence_band(
            self, rate, selective):
        r = channel_regime(self.DELAYS, self.POWERS, rate)
        assert r.frequency_selective is selective
        assert (r.signal_bandwidth_hz > r.coherence_bandwidth_hz) is selective

    @pytest.mark.parametrize('name', ['rms_delay_spread', 'energy_support',
                                      'coherence_bandwidth', 'channel_regime'])
    def test_the_arrivals_method_returns_what_the_function_does(self, name):
        a = _arrivals(self.DELAYS, self.AMPS)
        power = np.abs(a.received_amplitudes) ** 2
        args = (a.delays, power)
        method, func = getattr(a, name), globals()[name]
        if name == 'channel_regime':
            assert method(2000.0) == func(*args, 2000.0)
        else:
            assert method() == pytest.approx(func(*args), rel=1e-15)

    @pytest.mark.parametrize('name', ['rms_delay_spread', 'energy_support',
                                      'coherence_bandwidth', 'channel_regime'])
    def test_the_method_carries_no_arithmetic_of_its_own(self, name):
        body = _body(getattr(Arrivals, name))
        assert 'np.sqrt' not in body and 'np.cumsum' not in body
        assert 'ChannelRegime(' not in body

    def test_a_refusal_names_the_method_the_caller_called(self):
        with pytest.raises(ConfigurationError) as exc:
            _arrivals(self.DELAYS, self.AMPS).energy_support(0.0)
        assert str(exc.value).startswith('Arrivals.energy_support:')

    def test_complex_amplitudes_passed_as_powers_are_refused(self):
        """The one way to misuse the array-level form that a carrier could
        not: powers are |a|**2, so a negative entry means the caller passed
        amplitudes."""
        with pytest.raises(ConfigurationError, match='non-negative'):
            rms_delay_spread(self.DELAYS, -self.POWERS)


class TestTheArrivalSumHasOneImplementation:
    """``Arrivals.transfer_function`` and Bellhop's broadband synthesis were
    two spellings of the same sum, and disagreed on a negative amplitude:
    one took ``abs`` and the other did not."""

    F = np.linspace(100.0, 500.0, 9)

    def test_the_sum_matches_its_closed_form(self):
        H = arrival_transfer_function(
            self.F, [1.0, 0.6], [0.010, 0.030],
            delays_imag_s=[-1e-6, -3e-6], phases_rad=[0.0, np.pi])
        omega = 2 * np.pi * self.F
        closed = sum(A * np.exp(1j * ph) * np.exp(omega * ti)
                     * np.exp(-1j * omega * tr)
                     for A, ph, tr, ti in ((1.0, 0.0, .010, -1e-6),
                                           (0.6, np.pi, .030, -3e-6)))
        assert np.abs(H - closed).max() < 1e-28

    def test_absorption_lives_in_the_imaginary_delay_not_the_amplitude(self):
        """Its own reason for existing: the amplitude alone is lossless, so
        a band shows no absorption slope unless Im(tau) is applied."""
        flat = arrival_transfer_function(self.F, [1.0], [0.01])
        lossy = arrival_transfer_function(self.F, [1.0], [0.01],
                                          delays_imag_s=[-2e-5])
        assert np.ptp(np.abs(flat)) < 1e-12
        assert np.abs(lossy)[0] > np.abs(lossy)[-1]

    def test_the_arrivals_method_returns_what_the_function_does(self):
        a = _arrivals([0.010, 0.030], [1.0, 0.6],
                      imag=[-1e-6, -3e-6], phases=[0.0, np.pi])
        H = a.transfer_function(self.F)
        ref = arrival_transfer_function(
            self.F, [1.0, 0.6], [0.010, 0.030],
            delays_imag_s=[-1e-6, -3e-6], phases_rad=[0.0, np.pi])
        assert np.array_equal(np.asarray(H.data).ravel(), ref)

    def test_a_negative_amplitude_is_refused_rather_than_silently_halved(self):
        """The two paths disagreed by up to 10.7 dB per bin on this input,
        with no error from either. An amplitude is a magnitude; the sign
        belongs in the phase."""
        a = _arrivals([0.010, 0.030], [1.0, -0.6])
        with pytest.raises(ConfigurationError, match='negative'):
            a.transfer_function(self.F)
        with pytest.raises(ConfigurationError, match='negative'):
            arrival_transfer_function(self.F, [1.0, -0.6], [0.010, 0.030])

    def test_the_sign_is_expressible_the_way_the_refusal_says(self):
        """The refusal has to leave the caller a way through, so this pins
        that abs(A) with pi of phase is the same signal."""
        signed = arrival_transfer_function(self.F, [0.6], [0.03],
                                           phases_rad=[np.pi])
        assert np.abs(signed + arrival_transfer_function(
            self.F, [0.6], [0.03])).max() < 1e-15


class TestTheBroadbandLossIsReachableWithoutAField:
    """Ainslie Eq. 11.46 existed only inside ``Field.broadband_loss``."""

    F = np.linspace(100.0, 500.0, 21)

    def test_a_flat_unit_channel_loses_nothing(self):
        H = np.ones(self.F.size, dtype=complex)
        assert broadband_propagation_loss(H) == pytest.approx(0.0, abs=1e-12)

    def test_it_is_the_weighted_average_of_the_coherent_power(self):
        H = (np.linspace(0.2, 1.0, self.F.size)).astype(complex)
        w = np.linspace(1.0, 3.0, self.F.size)
        expected = 10 * np.log10(w.sum() / np.sum(w * np.abs(H) ** 2))
        assert broadband_propagation_loss(H, w) == pytest.approx(
            expected, rel=1e-13)

    def test_a_no_data_bin_is_carried_forward_not_averaged_away(self):
        H = np.ones(self.F.size, dtype=complex)
        H[3] = np.nan
        assert np.isnan(broadband_propagation_loss(H))

    def test_a_zero_weight_bin_still_carries_its_no_data_cell(self):
        """The subtle half of the line above: a bin weighted zero
        contributes nothing to the sum, so skipping it would let a NaN
        column average to a finite level."""
        H = np.ones(self.F.size, dtype=complex)
        H[3] = np.nan
        w = np.ones(self.F.size)
        w[3] = 0.0
        assert np.isnan(broadband_propagation_loss(H, w))

    def test_a_grid_transforms_in_one_call(self):
        rng = np.random.default_rng(3)
        block = (rng.normal(size=(2, 3, self.F.size))
                 + 1j * rng.normal(size=(2, 3, self.F.size)))
        out = broadband_propagation_loss(block)
        assert out.shape == (2, 3)
        for i in range(2):
            for j in range(3):
                assert out[i, j] == pytest.approx(
                    broadband_propagation_loss(block[i, j]), rel=1e-13)

    def test_the_field_method_carries_no_arithmetic_of_its_own(self):
        body = _body(Field.broadband_loss)
        assert 'broadband_propagation_loss(' in body
        assert 'np.log10' not in body


class TestTheResponseGateIsReachableWithoutAField:
    F = np.arange(0.0, 500.0, 5.0)      # 200 ms record

    def _two_paths(self):
        # 10 ms and 60 ms are exact multiples of the record's own step
        # (dt = 1/(N*df) = 2 ms); a delay between samples straddles two
        # bins and leaks across the gate.
        return (np.exp(-2j * np.pi * self.F * 0.010)
                + 0.8 * np.exp(-2j * np.pi * self.F * 0.060))

    def test_a_gate_narrower_than_the_separation_keeps_one_path(self):
        H = self._two_paths()
        kept = gate_transfer_function(H, self.F, 0.020)
        # One path left: a flat modulus, where two interfere into a ripple.
        assert np.ptp(np.abs(kept)) < 1e-9 < np.ptp(np.abs(H))

    def test_a_gate_wider_than_the_separation_keeps_both(self):
        H = self._two_paths()
        # The paths are 50 ms apart and the gate is centred on the louder.
        kept = gate_transfer_function(H, self.F, 0.060)
        assert np.abs(kept - H).max() < 1e-9

    def test_the_window_reaches_round_the_end_of_the_record(self):
        """The record wraps, so a path near its end sits next to one near
        its start. A linear distance would cut a path the gate should
        keep."""
        record = 1.0 / float(self.F[1] - self.F[0])
        H = (np.exp(-2j * np.pi * self.F * 0.002)
             + np.exp(-2j * np.pi * self.F * (record - 0.002)))
        kept = gate_transfer_function(H, self.F, 0.006, origin=0.0)
        assert np.abs(kept - H).max() < 1e-9

    def test_a_gate_that_would_keep_everything_is_refused(self):
        with pytest.raises(ConfigurationError, match='no-op'):
            gate_transfer_function(self._two_paths(), self.F, 0.5)

    def test_the_field_method_carries_no_arithmetic_of_its_own(self):
        body = _body(Field.truncate_response)
        assert 'gate_transfer_function(' in body
        assert 'np.fft.ifft' not in body and 'np.fft.fft' not in body


class TestTheLevelsAreReachableWithoutAField:
    def test_the_peak_level_is_the_peak_not_the_rms(self):
        x = np.array([0.0, 3e-4, -5e-4, 1e-4])
        assert peak_level(x) == pytest.approx(20 * np.log10(5e-4 / 1e-6))
        assert peak_level(x) > spl(x)

    def test_a_silent_record_floors_instead_of_reaching_minus_infinity(self):
        """Both level functions floor at the same constant, so a silent cell
        in a map does not poison a mean taken over it."""
        assert peak_level(np.zeros(8)) == pytest.approx(-180.0)
        assert sound_exposure_level(np.zeros(8), 1e-3) == pytest.approx(-180.0)
        assert np.isfinite(np.mean([peak_level(np.zeros(4)), 100.0]))

    def test_the_exposure_is_the_energy_not_the_mean_square(self):
        """SEL accumulates: doubling the duration of a steady signal adds
        3 dB here and nothing to spl."""
        fs = 1000.0
        one = np.sin(2 * np.pi * 100 * np.arange(1000) / fs)
        two = np.concatenate([one, one])
        gain = (sound_exposure_level(two, 1 / fs)
                - sound_exposure_level(one, 1 / fs))
        assert gain == pytest.approx(3.0103, abs=1e-3)
        assert spl(two) == pytest.approx(spl(one), abs=1e-9)

    def test_the_exposure_matches_the_canonical_conversion(self):
        fs = 1000.0
        p = np.sin(2 * np.pi * 100 * np.arange(512) / fs) * 1e-3
        assert sound_exposure_level(p, 1 / fs) == pytest.approx(
            power_to_dB(np.sum(p ** 2) / fs), rel=1e-15)

    def test_spl_reduces_along_a_named_axis(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 500)) * 1e-4
        assert np.abs(spl(x, axis=-1)
                      - np.array([spl(r) for r in x])).max() == 0.0

    @pytest.mark.parametrize('method,marker', [
        ('peak_sound_pressure_level', '_peak_level('),
        ('sound_exposure_level', '_sound_exposure_level('),
    ])
    def test_the_field_method_carries_no_arithmetic_of_its_own(self, method,
                                                               marker):
        body = _body(getattr(Field, method))
        assert marker in body
        assert 'np.log10' not in body


class TestTheModalComputationsAreReachableWithoutAModes:
    """A 433-line perturbation and the far-field modal sum were both method
    bodies. Both take plain ``k`` / ``psi`` arrays and nothing else."""

    Z = np.linspace(0.0, 100.0, 51)

    def _psi(self, n=4):
        return np.stack([np.sin((m + 1) * np.pi * self.Z / 100.0)
                         for m in range(n)], axis=1)

    def _modes(self, k):
        return Modes(k=k, phi=self._psi(), depths=self.Z, model='Kraken',
                     frequencies=100.0)

    def test_the_attenuation_matches_the_method(self):
        k = np.array([0.42, 0.40, 0.36, 0.30], dtype=complex)
        got = modal_attenuation(k, self._psi(), self.Z, 0.01, frequency=100.0)
        perturbed = self._modes(k).with_attenuation(alpha_dB_per_m=0.01)
        assert np.array_equal(np.asarray(perturbed.k).imag, got)
        # and it actually moved: an all-zero Im(k) would match a broken
        # integral just as well.
        assert np.all(got > 0.0)

    def test_the_attenuation_scales_with_the_volume_absorption(self):
        """A first-order perturbation is linear in alpha, which is the one
        thing a wrong integral would not reproduce."""
        k = np.array([0.42, 0.40], dtype=complex)
        psi = self._psi(2)
        a1 = modal_attenuation(k, psi, self.Z, 0.01, frequency=100.0)
        a2 = modal_attenuation(k, psi, self.Z, 0.02, frequency=100.0)
        assert np.allclose(a2, 2.0 * a1, rtol=1e-12)

    def test_a_mode_the_perturbation_cannot_solve_is_no_data(self):
        k = np.array([0.42, -0.1], dtype=complex)
        got = modal_attenuation(k, self._psi(2), self.Z, 0.01, frequency=100.0)
        assert np.isfinite(got[0]) and np.isnan(got[1])

    def test_a_missing_frequency_is_refused(self):
        with pytest.raises(ConfigurationError, match='frequency'):
            modal_attenuation(np.array([0.42 + 0j]), self._psi(1), self.Z,
                              0.01, frequency=0.0)

    def test_the_modal_sum_matches_the_method(self):
        k = np.array([0.42 - 0.001j, 0.40 + 0.002j, 0.36 - 5e-4j, 0.30 + 0j])
        m = self._modes(k)
        z_r = np.array([10.0, 40.0, 75.0])
        r = np.array([500.0, 2000.0, 5000.0])
        F = m.modal_propagation_loss(source_depth=25.0, receiver_depths=z_r,
                                     ranges_m=r)
        psi = self._psi()
        psi_zs = np.array([np.interp(25.0, self.Z, psi[:, i])
                           for i in range(psi.shape[1])])
        psi_zr = np.column_stack([np.interp(z_r, self.Z, psi[:, i])
                                  for i in range(psi.shape[1])])
        assert np.array_equal(np.asarray(F.data),
                              modal_field(k, psi_zs, psi_zr, r))

    def test_the_range_origin_is_no_data_not_a_large_number(self):
        k = np.array([0.42 + 0j])
        out = modal_field(k, np.array([1.0]), np.array([[1.0]]),
                          np.array([0.0, 100.0]))
        assert np.isnan(out[0, 0]) and np.isfinite(out[0, 1])

    def test_either_sign_of_the_imaginary_wavenumber_decays(self):
        """Kraken writes decay as Im k < 0 and ``with_attenuation`` as
        Im k > 0; a passive medium can only attenuate, so both must."""
        psi_zs, psi_zr = np.array([1.0]), np.array([[1.0]])
        r = np.array([100.0, 5000.0])
        for k in (np.array([0.42 + 0.001j]), np.array([0.42 - 0.001j])):
            p = modal_field(k, psi_zs, psi_zr, r)
            assert np.abs(p[0, 1]) < np.abs(p[0, 0])

    @pytest.mark.parametrize('method,marker', [
        ('with_attenuation', 'modal_attenuation('),
        ('modal_propagation_loss', 'modal_field('),
    ])
    def test_the_modes_method_carries_no_arithmetic_of_its_own(self, method,
                                                               marker):
        body = _body(getattr(Modes, method))
        assert marker in body
        assert 'np.trapezoid' not in body and 'np.einsum' not in body


class TestThePulseShapedTapsAreReachableWithoutArrivals:
    def test_a_single_arrival_on_a_symbol_instant_is_the_pulse_itself(self):
        taps, times = pulse_shaped_taps([1.0], [0.0], 1000.0, pulse='rc',
                                        rolloff=0.25, span=8)
        # A raised cosine is normalised to unit peak and samples to zero at
        # every other symbol instant, which is what Nyquist means here.
        assert np.abs(taps).max() == pytest.approx(1.0, rel=1e-9)
        on_instants = taps[np.abs(times * 1000.0
                                  - np.round(times * 1000.0)) < 1e-9]
        assert np.count_nonzero(np.abs(on_instants) > 1e-6) == 1

    def test_the_gains_are_carried_with_their_phase(self):
        taps, _ = pulse_shaped_taps([1j], [0.0], 1000.0, pulse='rc')
        assert np.abs(np.angle(taps[np.argmax(np.abs(taps))])
                      - np.pi / 2) < 1e-9

    @pytest.mark.parametrize('bad,match', [
        (dict(pulse='boxcar'), 'rc'),
        (dict(span=0), 'span'),
        (dict(symbol_rate=-1.0), 'symbol_rate'),
    ])
    def test_a_shape_it_cannot_place_is_refused(self, bad, match):
        kw = dict(gains=[1.0], delays_s=[0.0], symbol_rate=1000.0)
        kw.update(bad)
        with pytest.raises(ConfigurationError, match=match):
            pulse_shaped_taps(kw.pop('gains'), kw.pop('delays_s'),
                              kw.pop('symbol_rate'), **kw)

    def test_the_arrivals_branch_carries_no_placement_of_its_own(self):
        body = _body(Arrivals.channel_taps)
        assert 'pulse_shaped_taps(' in body
        assert 'rrc_pulse(' not in body


class TestTheArrayFactorIsTheSteeringVectorSeenFromTheSource:
    """The two agreed in modulus and could be a radian out in phase, and
    nothing stated the relation. Now the method IS the relation."""

    Z = np.array([10.0, 12.0, 14.0, 16.0, 21.0])
    ANGLES = np.linspace(-80.0, 80.0, 17)

    def _source(self, w):
        return Source(depths=self.Z, frequencies=np.array([300.0]), weights=w)

    def test_the_factor_matches_the_documented_identity(self):
        from uacpy.acoustic_signal import steering_vectors
        w = np.array([1.0, 0.8, 0.6, 0.8, 1.0])
        w = w / w.sum()
        af = self._source(w).array_factor(self.ANGLES, frequency=300.0,
                                          sound_speed=1500.0)
        identity = (np.sqrt(self.Z.size)
                    * np.conj(steering_vectors(self.Z - self.Z.mean(),
                                               self.ANGLES, 300.0, 1500.0)) @ w)
        assert np.abs(af - identity).max() == 0.0

    def test_the_naive_composition_agrees_in_level_and_not_in_phase(self):
        """Why the identity has to be written down: dropping the conjugate
        and the phase centre costs nothing a modulus test would see."""
        from uacpy.acoustic_signal import steering_vectors
        w = np.array([1.0, 0.8, 0.6, 0.8, 1.0])
        w = w / w.sum()
        af = self._source(w).array_factor(self.ANGLES, frequency=300.0,
                                          sound_speed=1500.0)
        naive = (np.sqrt(self.Z.size)
                 * steering_vectors(self.Z, self.ANGLES, 300.0, 1500.0) @ w)
        assert np.abs(np.abs(af) - np.abs(naive)).max() < 1e-12
        assert np.degrees(np.abs(np.angle(af / naive))).max() > 90.0


class TestTheGroupVelocityHasOneDifferenceQuotient:
    """``Modes.compute_group_velocity`` spelled out the same two-line
    difference as ``modal_group_velocity``, which its own docstring already
    named. The copies agreed numerically; their **guards** did not."""

    Z = np.linspace(0.0, 100.0, 21)

    def _pair(self, k_lo, k_hi, f=(50.0, 51.0)):
        psi = np.stack([np.sin((m + 1) * np.pi * self.Z / 100.0)
                        for m in range(len(k_lo))], axis=1)
        return (Modes(k=np.asarray(k_lo, dtype=complex), phi=psi,
                      depths=self.Z, model='Kraken', frequencies=f[0]),
                Modes(k=np.asarray(k_hi, dtype=complex), phi=psi,
                      depths=self.Z, model='Kraken', frequencies=f[1]))

    def test_the_method_returns_what_the_function_does(self):
        lo, hi = self._pair([0.40, 0.39, 0.37], [0.40210, 0.39230, 0.37280])
        assert np.array_equal(
            lo.compute_group_velocity(hi),
            modal_group_velocity(np.array([50.0, 51.0]),
                                 np.stack([np.asarray(lo.k),
                                           np.asarray(hi.k)]))[0])

    def test_the_pair_may_be_given_in_either_order(self):
        """The method takes two Modes, not an ordered grid, so it has to
        sort them before handing the function an increasing axis."""
        lo, hi = self._pair([0.40, 0.39, 0.37], [0.40210, 0.39230, 0.37280])
        assert np.allclose(lo.compute_group_velocity(hi),
                           hi.compute_group_velocity(lo), rtol=0, atol=0)

    def test_a_wavenumber_that_does_not_rise_is_refused_not_returned(self):
        """The behaviour delegation tightened. k_r rises with frequency
        because v_g is an energy-transport speed, so a falling step is bad
        input — it used to come back as a silently NEGATIVE speed, and a
        flat one as nan."""
        lo, hi = self._pair([0.40, 0.39], [0.39, 0.38])       # falling
        with pytest.raises(ConfigurationError):
            lo.compute_group_velocity(hi)
        flat, same = self._pair([0.40, 0.39], [0.40, 0.39])
        same = Modes(k=np.array([0.40, 0.39], dtype=complex), phi=flat.phi,
                     depths=self.Z, model='Kraken', frequencies=51.0)
        with pytest.raises(ConfigurationError):
            flat.compute_group_velocity(same)

    def test_two_frequencies_get_the_test_to_run_not_a_step_to_use(self):
        """A prescribed step is measured by decimating the grid, and a
        two-point grid has nothing to decimate — the notice must fall back
        to the advice that needs no grid. The method is always this case."""
        # A step of a known number of float32 spacings, so the floor it
        # reports is exact rather than approached.
        k0 = np.float32(0.35)
        kr = np.array([[k0], [k0 + np.float32(5e4) * np.spacing(k0)]],
                      dtype=np.float32).astype(float)
        with pytest.warns(UserWarning) as record:
            modal_group_velocity(np.array([100.0, 101.0]), kr)
        text = str(record[0].message)
        assert 'recompute at twice this frequency separation' in text
        # and NOT the prescribed-step form, which needs a grid to decimate
        assert 'decimat' not in text

    def test_the_method_carries_no_difference_quotient_of_its_own(self):
        body = _body(Modes.compute_group_velocity)
        assert 'modal_group_velocity(' in body
        assert 'np.errstate' not in body and 'domega' not in body


class TestTheTwoPublicTonesRoutesAgree:
    """``Field.extract_tone`` evaluates the transform AT the frequency;
    ``uacpy.io.rts_to_pressure`` sampled the nearest rfft bin. Both are
    public, both answer "the tone in this record", and they disagreed —
    reproducing the very numbers ``extract_tone``'s docstring records as
    the reason it was fixed. The fix had never reached the IO copy."""

    FS, NT = 1000.0, 512

    def _setup(self, frac):
        dt = 1.0 / self.FS
        t = np.arange(self.NT) * dt
        df = self.FS / self.NT
        f0 = 51 * df + frac * df          # bin 51 plus a fraction of one
        sig = 3.0 * np.cos(2 * np.pi * f0 * t + 0.4)
        return dt, t, f0, sig

    def _both(self, frac):
        from uacpy.io import rts_to_pressure
        dt, t, f0, sig = self._setup(frac)
        ranges = np.array([1000.0])
        got, _ = rts_to_pressure({"p": sig[:, None], "dt": dt,
                                  "ranges": ranges}, f0)
        trace = Field(data=sig.reshape(1, 1, -1),
                      coords={'depth': np.array([50.0]), 'range': ranges,
                              'time': t})
        ref = np.asarray(trace.extract_tone(f0).data).ravel()[0]
        return np.asarray(got).ravel()[0], ref

    @pytest.mark.parametrize('frac', [0.0, 0.1, 0.3, 0.5])
    def test_the_two_routes_now_agree_across_the_whole_bin(self, frac):
        got, ref = self._both(frac)
        assert abs(20 * np.log10(abs(got / ref))) < 1e-12
        assert abs(np.degrees(np.angle(got / ref))) < 1e-9

    def test_the_off_bin_tone_is_recovered_not_approximated(self):
        """Half a bin off, the nearest-bin answer was 0.85x the true
        amplitude with its phase 90 deg out. The estimator must return the
        tone it was given."""
        got, _ = self._both(0.5)
        # 1.7e-7 relative is the estimator's own floor, not slack: the
        # 2*X/sum(w) form assumes the negative-frequency image is clear of
        # the tone, and what is left of it through a Hann window is this.
        # Measured identical at amplitude 1 and 3, so it is relative.
        assert abs(got) == pytest.approx(3.0, rel=1e-6)
        assert abs(abs(got) / 3.0 - 1.0) < 1e-6
        assert np.degrees(np.angle(got)) == pytest.approx(
            np.degrees(0.4), abs=1e-4)

    def test_the_nearest_bin_answer_is_the_one_that_was_wrong(self):
        """The discriminator, not just an agreement check: this pins that
        the two estimators genuinely differ off a bin, so the test above
        cannot pass by both being wrong the same way."""
        dt, t, f0, sig = self._setup(0.5)
        win = np.hanning(self.NT)
        freqs = np.fft.rfftfreq(self.NT, dt)
        k = int(np.argmin(np.abs(freqs - f0)))
        nearest = 2.0 * np.fft.rfft(sig * win)[k] / np.sum(win)
        at = tone_phasor(sig, t, f0)
        assert 20 * np.log10(abs(nearest / at)) == pytest.approx(-1.418,
                                                                 abs=0.01)
        assert np.degrees(np.angle(nearest / at)) == pytest.approx(89.8,
                                                                   abs=0.1)

    def test_on_a_bin_it_reproduces_the_transform_it_replaces(self):
        dt, t, f0, sig = self._setup(0.0)
        win = np.hanning(self.NT)
        freqs = np.fft.rfftfreq(self.NT, dt)
        k = int(np.argmin(np.abs(freqs - f0)))
        nearest = 2.0 * np.fft.rfft(sig * win)[k] / np.sum(win)
        assert abs(tone_phasor(sig, t, f0) - nearest) < 1e-12

    def test_the_deconvolution_branch_is_exact_off_bin_too(self):
        """The ratio nearly cancels the bin error, which is why taking the
        bin looked defensible — and a CONSTANT channel cannot show that it
        does not, because there the ratio is exact at every offset by
        construction. Against a real three-path channel the bin ratio
        drifts; evaluating both sides at the frequency does not."""
        from uacpy.acoustic_signal.generate import sparc_pulse
        from uacpy.io import rts_to_pressure
        fs, nt = 1000.0, 512
        dt = 1.0 / fs
        t = np.arange(nt) * dt
        df = fs / nt
        freqs = np.fft.rfftfreq(nt, dt)

        def channel(f):
            return sum(a * np.exp(-2j * np.pi * f * tau)
                       for a, tau in ((1.0, 0.0), (0.6, 0.021),
                                      (0.35, 0.060)))

        for frac in (0.0, 0.25, 0.5):
            f0 = 51 * df + frac * df
            s_t, _ = sparc_pulse(t, 2 * np.pi * f0, 'P')
            p = np.fft.irfft(np.fft.rfft(s_t) * channel(freqs),
                             n=nt)[:, None]
            got, _ = rts_to_pressure(
                {"p": p, "dt": dt, "time": t, "ranges": np.array([1e3])},
                f0, pulse_type='P')
            ratio = np.asarray(got).ravel()[0] / channel(f0)
            assert abs(20 * np.log10(abs(ratio))) < 1e-6
            assert abs(np.degrees(np.angle(ratio))) < 1e-6

    def test_the_nearest_bin_ratio_is_what_that_test_discriminates(self):
        """The other half: a fixture on which the old and new answers agree
        proves nothing, so this pins that they genuinely differ. A constant
        channel is exactly such a fixture — hence the three-path one above."""
        fs, nt = 1000.0, 512
        dt = 1.0 / fs
        t = np.arange(nt) * dt
        df = fs / nt
        freqs = np.fft.rfftfreq(nt, dt)
        s_t = np.exp(-((t - 0.05) / 0.01) ** 2) * np.cos(2 * np.pi * 51 * df * t)

        def channel(f):
            return sum(a * np.exp(-2j * np.pi * f * tau)
                       for a, tau in ((1.0, 0.0), (0.6, 0.021),
                                      (0.35, 0.060)))

        f0 = 51 * df + 0.5 * df
        S = np.fft.rfft(s_t)
        p = np.fft.irfft(S * channel(freqs), n=nt)
        k = int(np.argmin(np.abs(freqs - f0)))
        bin_ratio = (np.fft.rfft(p)[k] / S[k]) / channel(f0)
        at_ratio = (tone_phasor(p, t, f0, window='none')
                    / tone_phasor(s_t, t, f0, window='none')) / channel(f0)
        assert np.degrees(np.angle(bin_ratio)) == pytest.approx(6.0, abs=0.5)
        assert abs(np.degrees(np.angle(at_ratio))) < 1e-9
        # and on a CONSTANT channel the two are indistinguishable
        const = 0.37 * np.exp(1j * 1.1)
        pc = np.fft.irfft(S * const, n=nt)
        flat = (np.fft.rfft(pc)[k] / S[k]) / const
        assert abs(np.degrees(np.angle(flat))) < 1e-12

    def test_neither_public_route_carries_the_estimator(self):
        import inspect
        from uacpy.io import oalib_reader
        for src in (_body(Field.extract_tone),
                    inspect.getsource(oalib_reader.rts_to_pressure)):
            assert 'tone_phasor(' in src
        assert 'np.sum(win)' not in _body(Field.extract_tone)


class TestTheCrossGridAgreementNumberIsObtainable:
    """``compare_models`` prints an RMS in every table cell, and
    ``metrics.tl_rmse`` *raises* on a grid mismatch. The interpolating
    version was private inside the plotter: you could see the number and
    not get it."""

    def _field(self, ranges, level):
        return Field(data=np.full((1, len(ranges)), level, dtype=complex),
                     coords={'depth': np.array([50.0]),
                             'range': np.asarray(ranges, dtype=float)},
                     frequencies=100.0)

    def test_an_aligned_pair_needs_no_resampling(self):
        from uacpy.metrics import tl_rmse, tl_rmse_on_shared_ranges
        r = np.linspace(1000.0, 5000.0, 9)
        a, b = self._field(r, 1e-3), self._field(r, 2e-3)
        assert tl_rmse_on_shared_ranges(a, b, depth=50.0) == pytest.approx(
            tl_rmse(a, b), rel=1e-12)

    def test_two_different_range_axes_are_compared_not_refused(self):
        from uacpy.metrics import tl_rmse, tl_rmse_on_shared_ranges
        a = self._field(np.linspace(1000.0, 5000.0, 9), 1e-3)
        b = self._field(np.linspace(1000.0, 5000.0, 17), 2e-3)
        with pytest.raises(ConfigurationError):
            tl_rmse(a, b)
        # 20*log10(2) = 6.02 dB apart everywhere, on either grid
        assert tl_rmse_on_shared_ranges(a, b, depth=50.0) == pytest.approx(
            20 * np.log10(2.0), rel=1e-9)

    def test_the_shared_grid_is_the_coarser_axis_either_way_round(self):
        """A symmetry the rule guarantees and 'always use the first
        field's axis' does not. It needs fields that VARY with range: two
        constants differ by the same amount on any grid, so they cannot
        tell the two rules apart — which is how my first version of this
        test passed under the mutation."""
        r_fine = np.linspace(1000.0, 5000.0, 65)
        r_coarse = np.linspace(1000.0, 5000.0, 9)
        fine = Field(
            data=(1e-3 * (1.0 + 0.7 * np.sin(r_fine / 300.0))
                  ).astype(complex).reshape(1, -1),
            coords={'depth': np.array([50.0]), 'range': r_fine},
            frequencies=100.0)
        coarse = Field(
            data=(2e-3 * np.ones_like(r_coarse)).astype(complex).reshape(1, -1),
            coords={'depth': np.array([50.0]), 'range': r_coarse},
            frequencies=100.0)
        from uacpy.metrics import tl_rmse_on_shared_ranges
        forward = tl_rmse_on_shared_ranges(fine, coarse, depth=50.0)
        backward = tl_rmse_on_shared_ranges(coarse, fine, depth=50.0)
        assert forward == pytest.approx(backward, rel=1e-12)
        # and the comparison is not trivially zero or nan
        assert np.isfinite(forward) and forward > 0.5

    def test_fields_sharing_no_range_are_no_data_not_agreement(self):
        from uacpy.metrics import tl_rmse_on_shared_ranges
        a = self._field(np.linspace(1000.0, 2000.0, 5), 1e-3)
        b = self._field(np.linspace(8000.0, 9000.0, 5), 1e-3)
        assert np.isnan(tl_rmse_on_shared_ranges(a, b, depth=50.0))

    def test_the_plotter_carries_no_arithmetic_of_its_own(self):
        import inspect
        from uacpy.visualization.plots import fields as F
        body = inspect.getsource(F._rms_between).split('"""')[2]
        assert 'tl_rmse_on_shared_ranges(' in body
        assert 'np.interp' not in body and 'np.sqrt' not in body


class TestTheWavenumberTransformIsReachable:
    """127 array-level lines implementing three of ``fieldsco.m``'s source
    geometries, private inside the ``.grn`` reader — while four
    user-facing docstrings cited them by that private path, naming an
    import no user could write."""

    def test_the_public_names_are_the_same_objects(self):
        from uacpy.core.acoustics import hankel_transform, wavenumber_taper
        from uacpy.io.grn_reader import _hankel_transform, _hanning_taper
        assert hankel_transform is _hankel_transform
        assert wavenumber_taper is _hanning_taper

    def test_a_kernel_from_anywhere_transforms(self):
        from uacpy.core.acoustics import hankel_transform
        k = np.linspace(0.02, 0.55, 128)
        G = np.ones((1, k.size), dtype=complex)
        out = hankel_transform(G, k, np.linspace(100.0, 2000.0, 5),
                               atten=1e-4)
        assert out.shape == (1, 5) and np.isfinite(out).all()

    def test_a_range_axis_past_the_alias_period_is_refused(self):
        """The guard that comes with it: beyond ~10/dk the wrapped tail is
        amplified by the stabilisation compensation."""
        from uacpy.core.acoustics import hankel_transform
        k = np.linspace(0.02, 0.55, 128)
        dk = float(k[1] - k[0])
        with pytest.raises(ConfigurationError, match='alias period'):
            hankel_transform(np.ones((1, k.size), dtype=complex), k,
                             np.array([20.0 / dk]), atten=1e-4)

    def test_the_taper_keeps_the_phase_speed_band_it_is_given(self):
        from uacpy.core.acoustics import wavenumber_taper
        k = np.linspace(0.02, 0.55, 256)
        flat = wavenumber_taper(k, 100.0, None, None)
        band = wavenumber_taper(k, 100.0, 1450.0, 1650.0)
        assert np.all(flat == 1.0)
        # ones across [w/cmax, w/cmin] and a Hanning roll-off to zero at
        # both ends of the grid — the window that decides which phase
        # speeds a spectral run keeps.
        omega = 2 * np.pi * 100.0
        inside = (k >= omega / 1650.0) & (k <= omega / 1450.0)
        assert inside.sum() > 10
        assert np.all(band[inside] == 1.0)
        assert band[0] == 0.0 and band[-1] == 0.0
        assert band.max() == 1.0

    def test_no_user_facing_docstring_cites_a_private_path(self):
        """The finding was not that the code was private but that the
        documentation named it anyway."""
        import inspect
        from uacpy.models import scooter, sparc
        for mod in (scooter, sparc):
            text = inspect.getsource(mod)
            assert 'grn_reader._hankel_transform' not in text
            assert 'grn_reader._hanning_taper' not in text


class TestTheCapabilitiesThatWereReachableOnlyThroughAModel:
    """Six computations that existed, worked, and could not be called on
    your own arrays. Each test states what a user can now do."""

    def test_a_complex_field_from_anywhere_converts_to_TL(self):
        from uacpy.core.acoustics import transmission_loss_dB
        from uacpy.core.results._base import _complex_to_dB
        p = np.array([1e-3, 1e-4, 0.0])
        got = transmission_loss_dB(p)
        assert got[0] == pytest.approx(60.0)
        assert got[1] == pytest.approx(80.0)
        # a cell no energy reached is capped, not +inf
        assert np.isfinite(got[2]) and got[2] == pytest.approx(600.0)
        # the in-package name is the same object, so the four callers and
        # the public one cannot drift
        assert transmission_loss_dB is _complex_to_dB

    def test_the_sign_is_a_loss_not_a_level(self):
        """The one way to misuse it: it is -20log10, so a quiet cell is a
        LARGE number, opposite to spl and power_to_dB."""
        from uacpy.core.acoustics import transmission_loss_dB, spl
        quiet, loud = np.array([1e-6]), np.array([1e-2])
        assert transmission_loss_dB(quiet)[0] > transmission_loss_dB(loud)[0]
        assert spl(quiet) < spl(loud)

    def test_a_waveform_spectrum_is_exact_off_the_dft_grid(self):
        from uacpy.acoustic_signal import waveform_spectrum_at
        fs, n = 4000.0, 256
        rng = np.random.default_rng(1)
        w = rng.normal(size=n)
        grid = np.fft.rfftfreq(n, 1 / fs)
        # on the waveform's own grid it reproduces rfft/fs
        assert np.abs(waveform_spectrum_at(w, fs, grid)
                      - np.fft.rfft(w) / fs).max() < 1e-12
        # off it, the direct DTFT — which interpolating the rfft is not
        off = grid[:-1] + 0.5 * (fs / n)
        t = np.arange(n) / fs
        dtft = np.array([np.sum(w * np.exp(-2j * np.pi * f * t)) / fs
                         for f in off])
        assert np.abs(waveform_spectrum_at(w, fs, off) - dtft).max() < 1e-12

    def test_a_non_uniform_request_takes_the_other_branch_and_agrees(self):
        """The function has TWO paths: a chirp-z contour for a uniform
        ascending run, and a direct outer product for anything else. Every
        other test here hands it a uniform grid, so the dense branch was
        unexercised — a mutation of its kernel went unnoticed."""
        from uacpy.acoustic_signal import waveform_spectrum_at
        fs, n = 4000.0, 192
        rng = np.random.default_rng(9)
        w = rng.normal(size=n)
        t = np.arange(n) / fs
        # deliberately NOT an arithmetic progression
        freqs = np.array([61.0, 137.5, 138.0, 900.25, 1501.0])
        dtft = np.array([np.sum(w * np.exp(-2j * np.pi * f * t)) / fs
                         for f in freqs])
        assert np.abs(waveform_spectrum_at(w, fs, freqs) - dtft).max() < 1e-12
        # and the chunked path, which splits the same sum into blocks
        chunked = waveform_spectrum_at(w, fs, freqs, _max_elems=64)
        assert np.abs(chunked - dtft).max() < 1e-12

    def test_out_of_band_frequencies_return_zero_not_an_alias(self):
        from uacpy.acoustic_signal import waveform_spectrum_at
        got = waveform_spectrum_at(np.ones(64), 1000.0,
                                   np.array([-10.0, 500.0, 900.0]))
        assert got[0] == 0.0 and got[2] == 0.0

    def test_a_wavenumber_step_reports_where_it_wraps(self):
        from uacpy.core.acoustics import (alias_period,
                                          ranges_fit_alias_period)
        dk = 4.4e-4
        assert alias_period(dk) == pytest.approx(2 * np.pi / dk)
        assert ranges_fit_alias_period(dk, 6_000.0)
        assert not ranges_fit_alias_period(dk, 20_000.0)
        # the boundary itself, both sides
        r = alias_period(dk)
        assert ranges_fit_alias_period(dk, np.nextafter(r, 0.0))
        assert not ranges_fit_alias_period(dk, r)

    def test_a_single_wavenumber_does_not_wrap(self):
        from uacpy.core.acoustics import (alias_period,
                                          ranges_fit_alias_period)
        assert alias_period(0.0) == float('inf')
        assert ranges_fit_alias_period(0.0, 1e9)

    def test_a_reception_carries_the_absorption_and_the_carrier(self):
        """What simulate_arrival_reception adds over simulate_reception:
        the exp(omega*Im tau) absorption and the carrier rotation."""
        from uacpy.acoustic_signal import simulate_arrival_reception
        fs, fc = 20000.0, 3000.0
        t = np.arange(400) / fs
        src = np.hanning(400) * np.sin(2 * np.pi * fc * t)
        lossless, _ = simulate_arrival_reception(
            src, [1.0, 1.0], [0.010, 0.030], fs, fc)
        lossy, _ = simulate_arrival_reception(
            src, [1.0, 1.0], [0.010, 0.030], fs, fc,
            delays_imag_s=[0.0, -5e-5])
        # the absorbed second arrival is quieter; the first is untouched
        assert np.abs(lossy).max() == pytest.approx(np.abs(lossless).max(),
                                                    rel=1e-9)
        assert np.sum(lossy ** 2) < 0.95 * np.sum(lossless ** 2)

    @pytest.mark.parametrize('bad,match', [
        (dict(amplitudes=[1.0, 0.5], delays_s=[0.01]), 'same length'),
        (dict(amplitudes=[1.0], delays_s=[0.01], phases_rad=[0.0, 1.0]),
         'phases_rad'),
    ])
    def test_a_mismatched_arrival_list_is_refused(self, bad, match):
        from uacpy.acoustic_signal import simulate_arrival_reception
        kw = dict(amplitudes=[1.0], delays_s=[0.01])
        kw.update(bad)
        with pytest.raises(ConfigurationError, match=match):
            simulate_arrival_reception(np.ones(32), sample_rate=8000.0,
                                       fc=1000.0, **kw)

    def test_the_bellhop_wrapper_carries_no_synthesis_of_its_own(self):
        import inspect
        from uacpy.models.bellhop import delayandsum
        body = inspect.getsource(delayandsum).split('"""')[2]
        assert 'simulate_arrival_reception(' in body
        assert 'hilbert' not in body and 'np.convolve' not in body

    def test_a_ram_user_can_ask_whether_their_grid_is_good_enough(self):
        """RAM's public surface was validate_inputs and select_backend; the
        grid answers reached the user only as warning strings."""
        from uacpy.models import (numerov_error, optimal_c0,
                                  rams_dz_shear_cap,
                                  rotated_pade_coefficients)
        # "What c0 should the PE expand about?"
        c0 = optimal_c0(1500.0, 1600.0, 30.0)
        assert 1400.0 <= c0 <= 1700.0
        # "Is this dz accurate at this frequency?" — and it must get WORSE
        # as the grid coarsens, which is the whole point of asking.
        k0 = 2 * np.pi * 50.0 / c0
        fine = numerov_error(0.5, k0, 30.0)
        coarse = numerov_error(4.0, k0, 30.0)
        assert np.isfinite(fine) and np.isfinite(coarse)
        assert coarse > fine
        # "How fine must dz be for this shear speed?"
        cap = rams_dz_shear_cap(300.0, 50.0)
        assert 0.0 < cap < 300.0 / 50.0
        # "What are the rotated Pade coefficients at this angle?"
        coeffs = rotated_pade_coefficients(6, 45.0)
        assert coeffs is not None


class TestTheTwoFunctionLevelDuplicates:
    """Neither was an instance of the method-wrapping rule; both were two
    spellings of one formula, and in one case the guards differed."""

    def test_the_sonar_background_sum_is_the_exported_one(self):
        from uacpy.sonar import total_reverberation
        a, b = np.array([60., 65., 70.]), np.array([62., 58., 71.])
        expected = 10 * np.log10(10 ** (a / 10) + 10 ** (b / 10))
        assert np.abs(total_reverberation(a, b) - expected).max() < 1e-12

    def test_the_signal_excess_carries_no_dB_sum_of_its_own(self):
        import inspect
        from uacpy.sonar import sonar_equation as SE
        body = inspect.getsource(SE.active_signal_excess)
        assert 'total_reverberation(' in body
        assert '10.0 ** (b / 10.0)' not in body

    def test_hamiltons_kp_has_one_implementation(self):
        """They agreed to 1.78e-15 everywhere a caller could reach, and
        differed 0.042 % past 9.5 phi where one returned the regression and
        the other its printed four-decimal value."""
        from uacpy.core.sediment import _hamilton_kp
        from uacpy.sonar.bottom_scattering import _grain_size_alpha_over_f
        for phi in np.linspace(-1.0, 11.0, 241):
            assert _grain_size_alpha_over_f(float(phi)) == _hamilton_kp(
                float(phi))
        # the branch that used to differ now gives the regression, not the
        # printed literal
        assert _grain_size_alpha_over_f(9.5) == pytest.approx(0.060075,
                                                              abs=1e-6)
