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
from uacpy.acoustic_signal.system import _MAX_DEFAULT_IR_SAMPLES
from uacpy.acoustic_signal import (
    channel_response,
    impulse_response,
    impulse_response_from_transfer_function,
    simulate_reception,
    transfer_function_from_impulse_response,
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
        from uacpy.acoustic_signal.system import impulse_response
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


class TestChannelResponseIsObtainableWithoutDrawingIt:
    """``h`` to ``H(f)`` is a public transform, not a step inside a plotter.

    ``plot_channel`` used to build the frequency panel itself, so the only
    way to get the channel's response with uacpy's conventions was to draw
    it and read the axes — and the two choices that shape the answer, the
    zero-padding rule and the dB floor, were buried in the plotter with no
    way to state either at the call.
    """

    FS = 4000.0

    @staticmethod
    def _taps(n=37, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_it_returns_the_two_sided_complex_response_centred_on_zero(self):
        f, H = channel_response(self._taps(), self.FS)
        assert np.iscomplexobj(H)
        assert f.size == H.size
        # Centred: the grid runs -fs/2 .. +fs/2, with 0 Hz on a bin.
        assert f[0] == pytest.approx(-self.FS / 2.0)
        assert np.any(f == 0.0)
        assert np.all(np.diff(f) > 0.0)

    def test_the_default_zero_padding_floors_a_short_response_at_1024(self):
        """Interpolation, not resolution: a 37-tap set still draws a curve."""
        assert channel_response(self._taps(37), self.FS)[0].size == 1024
        assert channel_response(self._taps(900), self.FS)[0].size == 1800

    def test_nfft_pins_the_grid(self):
        assert channel_response(self._taps(), self.FS, nfft=2048)[0].size == 2048

    def test_zero_padding_interpolates_rather_than_moving_the_bins(self):
        """The padded grid contains the unpadded one, value for value.

        The guarantee that makes the default safe: padding adds samples
        between the natural bins, it does not shift the ones already there.
        """
        h = self._taps(64)
        f_nat, H_nat = channel_response(h, self.FS, nfft=64)
        _f_pad, H_pad = channel_response(h, self.FS, nfft=128)
        assert np.allclose(H_pad[::2], H_nat, atol=1e-12)

    def test_a_real_response_comes_back_conjugate_symmetric(self):
        """A real ``h`` has ``H(-f) = conj(H(f))`` — the property the
        one-sided inverse relies on, measured rather than assumed."""
        f, H = channel_response(np.arange(1.0, 9.0), self.FS, nfft=64)
        # -fs/2 is on the grid and +fs/2 is not, so the Nyquist bin has no
        # partner to pair with: 32 negative frequencies against 31 positive.
        neg, pos = (f < 0) & (f > -self.FS / 2.0), f > 0
        assert H[neg].size == H[pos].size == 31
        assert np.allclose(H[neg], np.conj(H[pos][::-1]), atol=1e-12)

    def test_plot_channel_draws_exactly_what_the_transform_returns(self):
        """The plotter is a consumer: its curve is the transform in dB.

        Pinned by value, so extracting the computation cannot quietly
        change the picture the books and examples already show.
        """
        import matplotlib
        matplotlib.use('Agg')
        from uacpy.visualization.plots.comms import plot_channel
        h = self._taps()
        fig, ax = plot_channel(h, self.FS)
        try:
            drawn_f, drawn_db = ax[1].lines[0].get_data()
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
        f, H = channel_response(h, self.FS)
        assert np.array_equal(drawn_f, f)
        assert np.array_equal(drawn_db, 20 * np.log10(np.abs(H) + 1e-12))

    @pytest.mark.parametrize('h, fragment', [
        (np.zeros((2, 4)), '1-D impulse response'),
        (np.array([]), 'no channel to transform'),
    ])
    def test_it_refuses_an_input_that_is_not_one_channel(self, h, fragment):
        with pytest.raises(ConfigurationError, match=fragment):
            channel_response(h, 4000.0)

    def test_an_nfft_shorter_than_the_response_is_refused_not_truncated(self):
        """Truncation drops the tail rather than folding it, and nothing in
        the returned H says so — so it is refused at the call."""
        with pytest.raises(ConfigurationError, match='shorter than the 37-tap'):
            channel_response(self._taps(), self.FS, nfft=8)

    @pytest.mark.parametrize('rate', [0.0, -1.0, np.nan, np.inf])
    def test_an_unusable_sample_rate_is_refused(self, rate):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            channel_response(self._taps(), rate)


class TestTheTwoDirectionsShareOneConvention:
    """``h -> H`` must undo ``H -> h`` numerically, not up to a constant.

    ``impulse_response_from_transfer_function`` is a plain ``irfft``: no
    ``df``, no ``fs``. The first draft of the inverse divided by the sample
    rate to make ``H`` a spectral density, which is a defensible convention
    and the wrong one HERE — the pair came back a factor of ``fs`` apart
    with the phase still exact to 1e-16, so every phase test passed and
    nothing measured the level. These pin the ratio, which is what that
    mistake moved.
    """

    F = np.arange(0.0, 1000.0, 5.0)
    H = (np.exp(-2j * np.pi * F * 0.010)
         + 0.6 * np.exp(-2j * np.pi * F * 0.030))

    def test_the_round_trip_returns_the_same_amplitudes(self):
        _t, h = impulse_response_from_transfer_function(self.H, self.F, FS)
        f_back, H_back = transfer_function_from_impulse_response(
            h, FS, band=(self.F[0], self.F[-1]))
        ref = (np.interp(f_back, self.F, self.H.real)
               + 1j * np.interp(f_back, self.F, self.H.imag))
        # A scaling error shows up here and only here: the phases below
        # stay exact whatever constant multiplies H.
        assert np.abs(H_back).mean() / np.abs(ref).mean() == pytest.approx(
            1.0, abs=1e-9)
        assert np.abs(H_back - ref).max() < 1e-12

    def test_a_started_record_carries_its_offset_out_again(self):
        """``t0`` rotates the phase back; without it the modulus is right
        and the angle is not, which nothing but an interference notices."""
        fs, n = 2000.0, 512
        t0 = 0.125
        h = np.zeros(n)
        h[40] = 1.0
        f_rot, H_rot = transfer_function_from_impulse_response(h, fs, t0=t0)
        _f, H_raw = transfer_function_from_impulse_response(h, fs, t0=0.0)
        assert np.abs(np.abs(H_rot) - np.abs(H_raw)).max() < 1e-12
        expected = H_raw * np.exp(-2j * np.pi * f_rot * t0)
        assert np.abs(H_rot - expected).max() < 1e-12
        # The rotation is not a no-op at this offset.
        assert np.abs(H_rot - H_raw).max() > 1.0


class TestABlockOfResponsesTransformsInOneCall:
    """``axis=`` exists so a gridded result needs no Python loop — and so
    ``Field.to_transfer_function`` can delegate rather than carry a second
    copy of the transform."""

    RNG = np.random.default_rng(20260924)

    def _block(self):
        return self.RNG.normal(size=(2, 3, 64))

    def test_each_cell_matches_the_same_cell_transformed_alone(self):
        block = self._block()
        _f, H = transfer_function_from_impulse_response(block, FS, t0=0.02)
        for i in range(2):
            for j in range(3):
                _f1, H1 = transfer_function_from_impulse_response(
                    block[i, j], FS, t0=0.02)
                assert np.array_equal(H[i, j], H1)

    def test_the_time_axis_need_not_be_last(self):
        block = self._block()
        _f, last = transfer_function_from_impulse_response(block, FS)
        _f0, first = transfer_function_from_impulse_response(
            np.moveaxis(block, -1, 0), FS, axis=0)
        assert np.array_equal(np.moveaxis(first, 0, -1), last)

    @pytest.mark.parametrize('axis', [3, -4, 't', None])
    def test_an_axis_the_array_does_not_have_is_refused(self, axis):
        with pytest.raises(ConfigurationError) as exc:
            transfer_function_from_impulse_response(
                self._block(), FS, axis=axis)
        assert 'axis' in str(exc.value)

    def test_one_sample_along_the_named_axis_is_refused(self):
        with pytest.raises(ConfigurationError) as exc:
            transfer_function_from_impulse_response(
                np.zeros((2, 3, 1)), FS)
        assert 'two samples' in str(exc.value)


class TestTheBandEdgeSurvivesFloatingPointDust:
    """A band taken from the grid that made ``h`` lands ON a bin, and a
    bare ``<=`` then keeps or drops that bin depending on the last bit:
    re-inverting a sample rate moved a 995 Hz edge by 1e-13 Hz and cost a
    bin, which is a silent off-by-one in the returned spectrum."""

    def test_an_edge_bin_displaced_by_rounding_is_still_kept(self):
        fs, n = 5120.0, 1024
        h = np.zeros(n)
        h[3] = 1.0
        full, _H = transfer_function_from_impulse_response(h, fs)
        high = float(full[199])            # 995 Hz, exactly a bin
        nudged = np.nextafter(high, 0.0)   # the same bin, one ulp low
        kept, _ = transfer_function_from_impulse_response(
            h, fs, band=(100.0, nudged))
        assert kept.size == 180
        assert kept[-1] == high

    def test_an_edge_genuinely_below_a_bin_still_excludes_it(self):
        """The other side of the same threshold. Asking for one bin less
        is not a boundary test — it lands half a bin from the tolerance
        and passes however wide the tolerance is. This asks for an edge
        that misses the bin by a millionth of a spacing: a thousand times
        the tolerance, and still a millionth of the gap to the neighbour,
        so only a tolerance that has stopped being dust keeps it."""
        fs, n = 5120.0, 1024
        h = np.zeros(n)
        h[3] = 1.0
        full, _H = transfer_function_from_impulse_response(h, fs)
        df = float(full[1] - full[0])
        kept, _ = transfer_function_from_impulse_response(
            h, fs, band=(100.0, float(full[199]) - 1e-6 * df))
        assert kept.size == 179
        assert kept[-1] == pytest.approx(float(full[198]))


class TestFieldToTransferFunctionDelegates:
    """The Field method is the array function plus axis bookkeeping, a
    band read off the identity, and the ``dt`` that carries the result
    into the density convention ``to_time_trace`` produces. What it must
    NOT be is a second implementation of the transform."""

    def test_the_method_carries_no_fft_of_its_own(self):
        import inspect
        from uacpy.core.results.field import Field
        body = inspect.getsource(Field.to_transfer_function)
        body = body.split('"""')[2]        # past the docstring
        assert 'transfer_function_from_impulse_response(' in body
        assert 'np.fft.rfft' not in body

    def test_a_refusal_names_the_method_the_caller_called(self):
        """Delegation must not surface a function name the user never
        typed: `who=` carries the caller's name into the message."""
        from uacpy.core.results.field import Field
        t = np.arange(256) / FS
        field = Field(data=np.zeros((1, 1, t.size)),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([1000.0]), 'time': t})
        with pytest.raises(ConfigurationError) as exc:
            field.to_transfer_function(band=(9e3, 1e4))
        assert str(exc.value).startswith('Field.to_transfer_function:')

    def test_a_gridded_field_matches_the_array_function_cell_by_cell(self):
        from uacpy.core.results.field import Field
        rng = np.random.default_rng(7)
        t = np.arange(256) / FS
        data = rng.normal(size=(2, 3, t.size))
        field = Field(data=data,
                      coords={'depth': np.array([10.0, 20.0]),
                              'range': np.array([1e3, 2e3, 3e3]),
                              'time': t})
        H = field.to_transfer_function()
        for i in range(2):
            for j in range(3):
                _f, ref = transfer_function_from_impulse_response(
                    data[i, j], FS)
                # dt is the one thing the method adds to the transform.
                assert np.abs(H.data[i, j] - ref / FS).max() < 1e-18
