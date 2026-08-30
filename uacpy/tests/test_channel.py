"""Tests for ``uacpy.acoustic_signal``'s channel model.

The impulse response a multipath channel presents to a signal: where the taps
land, how a fractional delay is interpolated, and what a transfer function
becomes when it is turned back into time.

Three things here are guards rather than behaviour, and each closes a way the
channel could answer a question it cannot actually answer:

* a delay grid finer than the sample rate can resolve is placed with a
  windowed-sinc kernel, not silently rounded to the nearest sample;
* a band lying entirely above Nyquist is refused instead of returning zeros,
  and one that straddles Nyquist warns and keeps the part below it;
* a tap count is bounded, so an arrival list that cannot be represented on the
  requested grid is reported rather than truncated in silence.
"""

import warnings

import numpy as np
import pytest

from uacpy import comms
from uacpy.acoustic_signal.channel import _MAX_DEFAULT_IR_SAMPLES
from uacpy.acoustic_signal import (
    impulse_response,
    impulse_response_from_transfer_function,
    simulate_reception,
)
from uacpy.core.exceptions import ConfigurationError

FS = 10000.0

#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]


class TestChannel:
    def test_integer_delays_place_taps(self):
        t, h = impulse_response([1.0, 0.5], [0.01, 0.02], FS, fractional=False)
        assert h[100] == pytest.approx(1.0)
        assert h[200] == pytest.approx(0.5)

    def test_fractional_splits_energy(self):
        # Delay 0.0105 s = sample 105.0 exactly -> single tap even when fractional.
        _, h = impulse_response([1.0], [0.0105], FS, fractional=True)
        assert h[105] == pytest.approx(1.0)
        # A genuinely fractional delay is placed with a windowed-sinc kernel,
        # normalised to unit DC gain. It is deliberately NOT two taps: a
        # two-tap linear split is a frac-dependent lowpass (-3.0 dB at
        # f/fs = 0.25 for frac = 0.5), not a fractional delay.
        _, h2 = impulse_response([1.0], [0.01005], FS, fractional=True)
        assert h2.sum() == pytest.approx(1.0)
        assert np.count_nonzero(h2) > 2

    def test_simulate_reception_shifts_transmit(self):
        tx = np.array([1.0, -1.0, 0.5])
        t, rx = simulate_reception(tx, [1.0], [0.01], FS)
        assert np.allclose(rx[100:103], tx)

    def test_ir_from_flat_transfer_function_is_delta(self):
        f = np.linspace(0, FS / 2, 65)
        H = np.ones_like(f, dtype=complex)
        _, h = impulse_response_from_transfer_function(H, f, FS, n_samples=128)
        assert np.argmax(np.abs(h)) == 0

    def test_ir_from_bandlimited_tf_has_no_out_of_band_energy(self):
        # H given only on a band; out-of-band DFT bins must be zero, not
        # held at the band-edge values (constant extrapolation would put an
        # artificial DC-to-band plateau into the impulse response).
        f = np.linspace(1000.0, 2000.0, 41)
        H = np.ones_like(f, dtype=complex)
        n = 256
        _, h = impulse_response_from_transfer_function(H, f, FS, n_samples=n)
        spec = np.fft.rfft(h, n=n)
        grid = np.fft.rfftfreq(n, 1.0 / FS)
        out_band = (grid < 900.0) | (grid > 2100.0)
        in_band = (grid >= 1100.0) & (grid <= 1900.0)
        assert np.max(np.abs(spec[out_band])) < 1e-9 * np.max(np.abs(spec))
        assert np.min(np.abs(spec[in_band])) > 0.5

    def test_negative_delay_raises(self):
        with pytest.raises(ConfigurationError):
            impulse_response([1.0], [-0.01], FS)


class TestFractionalDelayIsFlat:
    """A two-tap linear split is not a fractional delay: its response
    ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose attenuation depends on
    ``frac``, with a full null at Nyquist for ``frac = 0.5`` — measured
    -3.010 dB at ``f/fs = 0.25`` and -10.192 dB at 0.40. Two arrivals a
    propagation model reports as equal came back differing by up to 10 dB,
    decided by the sub-sample part of their travel times.

    Peak amplitude is deliberately *not* the metric here: a unit impulse at a
    fractional delay genuinely has no unit sample (the band-limited truth
    ``sinc(n-p)`` peaks at 0.900 for frac=0.25), so a peak test would overstate
    the defect. The spectral flatness is what is wrong and what is fixed."""

    FS = 40000.0

    def _kernel_response(self, frac, n=256):
        from uacpy.acoustic_signal import impulse_response
        _, h = impulse_response([1.0], [(100 + frac) / self.FS], self.FS,
                                n_samples=n)
        return np.fft.rfftfreq(n), np.fft.rfft(h)

    @pytest.mark.parametrize('frac', [0.25, 0.5, 0.75])
    def test_response_is_flat_across_the_band(self, frac):
        f, H = self._kernel_response(frac)
        band = f <= 0.35
        assert np.max(np.abs(20 * np.log10(np.abs(H[band])))) < 0.1

    @pytest.mark.parametrize('frac', [0.25, 0.5])
    def test_group_delay_is_exact(self, frac):
        # Group delay was correct before too — the defect was amplitude only,
        # which is why it was silent. It must stay correct.
        f, H = self._kernel_response(frac)
        ph = np.unwrap(np.angle(H))
        i = int(np.argmin(np.abs(f - 0.2)))
        gd = -(ph[i + 1] - ph[i - 1]) / (2 * np.pi * (f[i + 1] - f[i - 1]))
        assert gd == pytest.approx(100.0 + frac, abs=1e-3)

    def test_truncated_kernel_warns_instead_of_dumping_full_amplitude(self):
        # An arrival at 10.9 samples used to land entirely at sample 10 at
        # full amplitude, 0.9 samples early and silently.
        from uacpy.acoustic_signal import impulse_response
        with pytest.warns(UserWarning, match='truncated'):
            _, h = impulse_response([1.0], [10.9 / self.FS], self.FS,
                                    n_samples=11)
        assert h[10] < 0.2

    def test_integer_delays_are_untouched(self):
        # The discriminating counterpart: an arrival exactly on a sample must
        # stay a clean unit impulse, and must not warn.
        from uacpy.acoustic_signal import impulse_response
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, h = impulse_response([1.0], [100.0 / self.FS], self.FS,
                                    n_samples=256)
        assert h[100] == pytest.approx(1.0)
        assert np.max(np.abs(np.delete(h, 100))) < 1e-12


def test_nearest_sample_placement_rounds():
    """``fractional=False`` promises *nearest*-sample placement; floor
    placement puts a 0.7-sample arrival one tap early."""
    from uacpy.acoustic_signal import impulse_response
    fs = 8000.0
    _, h = impulse_response([1.0], [0.7 / fs], fs, fractional=False)
    assert np.argmax(np.abs(h)) == 1
    # fractional=True uses a windowed-sinc kernel, not a two-tap linear
    # split: the latter is a frac-dependent lowpass (-3.0 dB at
    # f/fs = 0.25 for frac = 0.5), so equal arrivals came back unequal.
    # Placed clear of the array ends, where the kernel is not truncated.
    _, hf = impulse_response([1.0], [100.7 / fs], fs, fractional=True,
                             n_samples=256)
    assert hf.sum() == pytest.approx(1.0)
    centroid = float(np.sum(np.arange(hf.size) * hf) / hf.sum())
    assert centroid == pytest.approx(100.7, abs=0.01)


class TestTruncatedArrivalWarningNamesBothDirections:
    """``impulse_response`` warns when a fractional arrival sits too close to
    an end for its interpolation kernel to fit. The kernel's dropped sinc tail
    sums either way, so the amplitude error is not one-directional and the
    worst case is a GAIN: measured DC gain 1.1274 (+1.04 dB) for an arrival
    0.5 samples from the start against 0.9862 at 3.5 samples, at fs = 20 kHz
    over 128 samples. The warning previously said such arrivals "lose
    amplitude", which mis-directs the diagnosis.
    """

    FS, N = 20000.0, 128

    def _run(self, position_samples):
        import warnings as _w
        from uacpy.acoustic_signal.channel import impulse_response
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            _t, h = impulse_response(np.array([1.0]),
                                     np.array([position_samples / self.FS]),
                                     self.FS, n_samples=self.N)
        return h, [str(x.message) for x in caught
                   if 'interpolation kernel' in str(x.message)]

    def test_a_truncated_arrival_can_come_back_louder(self):
        h, msgs = self._run(0.5)
        assert msgs, 'the truncation warning did not fire'
        assert float(np.sum(np.real(h))) > 1.0, (
            'this case gains amplitude, so the warning must not claim a loss')

    def test_the_warning_does_not_claim_a_direction_it_lacks(self):
        _h, msgs = self._run(0.5)
        assert 'lose amplitude' not in msgs[0]
        assert 'either direction' in msgs[0]

    def test_an_arrival_clear_of_the_ends_is_exact_and_silent(self):
        h, msgs = self._run(20.0)
        assert not msgs
        assert float(np.sum(np.real(h))) == pytest.approx(1.0, abs=1e-9)


class TestImpulseResponseTapBound:
    def test_hour_long_delay_at_96k_raises_before_allocating(self):
        # 3600 s * 96 kHz = 3.456e8 taps (5.5 GB complex128); the bound
        # raises on the arithmetic, before np.zeros runs.
        with pytest.raises(ConfigurationError, match="3600.*96000"):
            impulse_response([1.0], [3600.0], 96000.0)

    @pytest.mark.parametrize("delay", [np.inf, np.nan])
    def test_nonfinite_delay_raises_typed(self, delay):
        with pytest.raises(ConfigurationError, match="default limit"):
            impulse_response([1.0], [delay], 8000.0)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_nonpositive_or_nonfinite_sample_rate_raises(self, bad):
        with pytest.raises(ConfigurationError,
                           match="sample_rate must be > 0 Hz and finite"):
            impulse_response([1.0], [0.1], bad)

    def test_explicit_n_samples_is_used_as_given(self):
        t, h = impulse_response([1.0], [0.004], 8000.0, n_samples=64)
        assert t.size == 64 and h.size == 64

    def test_short_delay_places_the_arrival(self):
        _, h = impulse_response([1.0], [0.01], 1000.0)
        assert int(np.argmax(np.abs(h))) == 10

    def test_multipath_channel_inherits_the_tap_bound(self):
        with pytest.raises(ConfigurationError, match="default limit"):
            comms.multipath_channel([1.0], [3600.0], 96000.0)


class TestImpulseResponseReportsBandsOutsideNyquist:
    FS = 10000.0

    def _flat(self, f0, f1):
        f = np.arange(f0, f1 + 1.0, 1.0)
        return f, np.ones(f.size, dtype=complex)

    def test_band_entirely_above_nyquist_raises_instead_of_returning_zeros(self):
        f, H = self._flat(6000.0, 6100.0)
        with pytest.raises(ConfigurationError, match="entirely above the Nyquist"):
            impulse_response_from_transfer_function(H, f, self.FS)

    def test_band_straddling_nyquist_warns_and_keeps_the_part_below(self):
        f, H = self._flat(4950.0, 5050.0)
        with pytest.warns(UserWarning, match="dropped from h"):
            _t, h = impulse_response_from_transfer_function(H, f, self.FS)
        f_in, H_in = self._flat(1000.0, 1100.0)
        _t, h_in = impulse_response_from_transfer_function(H_in, f_in, self.FS)
        # Half the band is lost, so the peak is about half the in-band one.
        assert np.max(np.abs(h)) == pytest.approx(0.5 * np.max(np.abs(h_in)),
                                                  rel=0.05)

    def test_band_reaching_exactly_nyquist_is_silent(self):
        f = np.linspace(0.0, self.FS / 2, 65)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _t, h = impulse_response_from_transfer_function(
                np.ones(f.size, dtype=complex), f, self.FS, n_samples=128)
        assert np.argmax(np.abs(h)) == 0


class TestTransferFunctionIRGridSpacing:
    """The default DFT grid of ``impulse_response_from_transfer_function``
    follows the spacing of ``frequencies``, so the documented unambiguous
    delay window ``1/df`` holds for a band-limited ``H(f)`` too. Sizing it as
    ``2*(f.size - 1)`` instead agreed with the docstring only when
    ``frequencies`` spanned all of ``[0, fs/2]``; on a 100-200 Hz band at 1 Hz
    spacing it gave df = 50 Hz — a 20 ms window that wrapped a 30 ms arrival
    onto 10 ms with nothing in the output to show for it."""

    FS = 10000.0

    def _delay_peak_ms(self, tau_s, frequencies):
        H = np.exp(-2j * np.pi * frequencies * tau_s)
        t, h = impulse_response_from_transfer_function(H, frequencies, self.FS)
        return t[int(np.argmax(np.abs(h)))] * 1e3

    @pytest.mark.parametrize("tau_ms", [5.0, 30.0, 55.0])
    def test_bandlimited_h_resolves_delays_out_to_one_over_the_spacing(
            self, tau_ms):
        f = np.arange(100.0, 201.0, 1.0)
        assert self._delay_peak_ms(tau_ms * 1e-3, f) == pytest.approx(tau_ms,
                                                                     abs=0.2)

    def test_default_grid_is_sample_rate_over_spacing(self):
        f = np.arange(100.0, 201.0, 1.0)
        _, h = impulse_response_from_transfer_function(
            np.ones(f.size, dtype=complex), f, self.FS)
        assert h.size == int(round(self.FS / 1.0))

    def test_full_band_input_recovers_its_own_length(self):
        # The grid a full-band rfftfreq came from is returned unchanged.
        for n in (128, 129, 256):
            grid = np.fft.rfftfreq(n, 1.0 / self.FS)
            _, h = impulse_response_from_transfer_function(
                np.ones(grid.size, dtype=complex), grid, self.FS)
            assert h.size == n

    def test_absurdly_fine_spacing_raises_instead_of_allocating(self):
        f = np.array([0.0, 1e-4, 2e-4])
        with pytest.raises(ConfigurationError, match="n_samples"):
            impulse_response_from_transfer_function(
                np.ones(3, dtype=complex), f, 1e6)
        # The limit is a default-only guard: an explicit n_samples is obeyed.
        _, h = impulse_response_from_transfer_function(
            np.ones(3, dtype=complex), f, 1e6, n_samples=64)
        assert h.size == 64
        assert _MAX_DEFAULT_IR_SAMPLES > 0


class TestImpulseResponseDropWarnings:
    """Every path that drops an arrival in its entirety says so: the
    quantised (``fractional=False``) path, the fractional path with an
    integer delay, and the fractional path with no kernel tap in the
    window. In-window arrivals are placed silently."""

    FS, N = 1000.0, 10

    def _run(self, delay_samples, fractional):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _t, h = impulse_response([1.0], [delay_samples / self.FS],
                                     self.FS, n_samples=self.N,
                                     fractional=fractional)
        return h, [str(w.message) for w in caught]

    def test_quantised_path_out_of_window_arrival_warns(self):
        h, msgs = self._run(20.0, fractional=False)
        assert any('lie entirely outside' in m for m in msgs)
        assert float(np.sum(np.abs(h))) == 0.0

    def test_fractional_path_integer_delay_out_of_window_warns(self):
        h, msgs = self._run(20.0, fractional=True)
        assert any('lie entirely outside' in m for m in msgs)
        assert float(np.sum(np.abs(h))) == 0.0

    def test_fractional_arrival_with_no_tap_in_window_reports_a_drop(self):
        _h, msgs = self._run(20.5, fractional=True)
        assert any('lie entirely outside' in m for m in msgs)
        assert not any('truncated' in m for m in msgs)

    @pytest.mark.parametrize('fractional', [True, False])
    def test_in_window_arrivals_are_silent(self, fractional):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _t, h = impulse_response([1.0], [5.0 / self.FS], self.FS,
                                     n_samples=self.N,
                                     fractional=fractional)
        assert float(np.sum(np.abs(h))) == pytest.approx(1.0, rel=1e-9)

    def test_truncation_advice_pairs_fractional_false_with_its_drop_warning(self):
        """The clipped-kernel warning's alternative (``fractional=False``)
        is described together with its own drop warning, so following the
        advice cannot silence a dropped arrival."""
        _h, msgs = self._run(9.9, fractional=True)
        clipped = next(m for m in msgs if 'truncated' in m)
        assert 'quantise instead' not in clipped
        assert 'dropped arrival' in clipped
