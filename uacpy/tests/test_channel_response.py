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

import re
import warnings

import numpy as np
import pytest

from uacpy import comms
from uacpy.acoustic_signal.channel import _MAX_DEFAULT_IR_SAMPLES
from uacpy.acoustic_signal import (
    arrival_grid_transfer_function,
    channel_response,
    impulse_response,
    impulse_response_from_transfer_function,
    simulate_reception,
    transfer_function_from_impulse_response,
)
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.tests.conftest import recorded_warnings
from uacpy.tests.conftest import warning_messages

FS = 10000.0

#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]


def _cell_tf(cell, frequencies, **kwargs):
    """``H(f)`` of one Bellhop arrival cell
    (:func:`~uacpy.acoustic_signal.arrival_grid_transfer_function`)."""
    return arrival_grid_transfer_function(frequencies, [[cell]],
                                          **kwargs)[0, 0]


class TestChannel:
    def test_integer_delays_place_taps(self):
        t, h = impulse_response([1.0, 0.5], [0.01, 0.02], sample_rate=FS, fractional=False)
        assert h[100] == pytest.approx(1.0)
        assert h[200] == pytest.approx(0.5)

    def test_fractional_splits_energy(self):
        # Delay 0.0105 s = sample 105.0 exactly -> single tap even when fractional.
        _, h = impulse_response([1.0], [0.0105], sample_rate=FS, fractional=True)
        assert h[105] == pytest.approx(1.0)
        # A genuinely fractional delay is placed with a windowed-sinc kernel,
        # normalised to unit DC gain. It is deliberately NOT two taps: a
        # two-tap linear split is a frac-dependent lowpass (-3.0 dB at
        # f/fs = 0.25 for frac = 0.5), not a fractional delay.
        _, h2 = impulse_response([1.0], [0.01005], sample_rate=FS, fractional=True)
        assert h2.sum() == pytest.approx(1.0)
        assert np.count_nonzero(h2) > 2

    def test_simulate_reception_shifts_transmit(self):
        tx = np.array([1.0, -1.0, 0.5])
        t, rx = simulate_reception([1.0], [0.01], source_waveform=tx,
                                   sample_rate=FS)
        assert np.allclose(rx[100:103], tx)

    def test_simulate_reception_refuses_complex_taps_on_a_real_transmit(self):
        """A real waveform times ``e^{i pi/2}`` is not a rotated waveform:
        its real part is zero where the reception is the quadrature copy.
        The analytic route rotates it. Measured against the exact quarter
        turn (``X·H`` on a 2**16-sample record): the analytic route 3.8e-5 of
        the peak off — ``analytic_signal`` is the waveform's own circular
        Hilbert transform over its 192 samples — and
        simulate_arrival_reception, the band-limited taps of the arrival's
        transfer function, 5.5e-7."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal import (analytic_signal,
                                           simulate_arrival_reception)
        n = 192
        tx = np.sin(2 * np.pi * 2000.0 * np.arange(n) / FS) * np.hanning(n)
        with pytest.raises(ConfigurationError, match="source_waveform is real"):
            simulate_reception([1j], [0.01], source_waveform=tx,
                               sample_rate=FS)
        _, rx = simulate_reception([1j], [0.01],
                                   source_waveform=analytic_signal(tx),
                                   sample_rate=FS)
        _, ref = simulate_arrival_reception([1.0], [0.01], source_waveform=tx,
                                            sample_rate=FS,
                                            trace_frequency=2000.0,
                                            phases_rad=[np.pi / 2],
                                            t_start=0.0)
        from uacpy.acoustic_signal import arrival_transfer_function
        L = 1 << 16
        f = np.fft.rfftfreq(L, 1.0 / FS)
        H = arrival_transfer_function(f, [1.0], [0.01], phases_rad=[np.pi / 2])
        H[0], H[-1] = H[0].real, H[-1].real
        exact = np.fft.irfft(np.fft.rfft(tx, L) * H, L)
        i0 = int(round(0.01 * FS))
        window = slice(i0, i0 + n)
        peak = np.abs(exact).max()
        assert np.max(np.abs(ref[window] - exact[window])) < 1e-6 * peak
        assert np.max(np.abs(np.real(rx)[window] - exact[window])) < 5e-5 * peak

    def test_ir_from_flat_transfer_function_is_delta(self):
        f = np.linspace(0, FS / 2, 65)
        H = np.ones_like(f, dtype=complex)
        _, h = impulse_response_from_transfer_function(H, frequencies=f, sample_rate=FS, n_samples=128)
        assert np.argmax(np.abs(h)) == 0

    def test_ir_from_bandlimited_tf_has_no_out_of_band_energy(self):
        # H given only on a band; out-of-band DFT bins must be zero, not
        # held at the band-edge values (constant extrapolation would put an
        # artificial DC-to-band plateau into the impulse response).
        f = np.linspace(1000.0, 2000.0, 41)
        H = np.ones_like(f, dtype=complex)
        _, h = impulse_response_from_transfer_function(H, frequencies=f, sample_rate=FS)
        n = h.size
        assert n == round(FS / (f[1] - f[0]))
        spec = np.fft.rfft(h, n=n)
        grid = np.fft.rfftfreq(n, 1.0 / FS)
        out_band = (grid < 900.0) | (grid > 2100.0)
        in_band = (grid >= 1100.0) & (grid <= 1900.0)
        assert np.max(np.abs(spec[out_band])) < 1e-9 * np.max(np.abs(spec))
        assert np.min(np.abs(spec[in_band])) > 0.5

    def test_negative_delay_raises(self):
        with pytest.raises(ConfigurationError, match='delays_s must be >= 0'):
            impulse_response([1.0], [-0.01], sample_rate=FS)


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
        _, h = impulse_response([1.0], [(100 + frac) / self.FS], sample_rate=self.FS,
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
            _, h = impulse_response([1.0], [10.9 / self.FS], sample_rate=self.FS,
                                    n_samples=11)
        assert h[10] < 0.2

    def test_integer_delays_are_untouched(self):
        # The discriminating counterpart: an arrival exactly on a sample must
        # stay a clean unit impulse, and must not warn.
        from uacpy.acoustic_signal import impulse_response
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, h = impulse_response([1.0], [100.0 / self.FS], sample_rate=self.FS,
                                    n_samples=256)
        assert h[100] == pytest.approx(1.0)
        assert np.max(np.abs(np.delete(h, 100))) < 1e-12


def test_nearest_sample_placement_rounds():
    """``fractional=False`` promises *nearest*-sample placement; floor
    placement puts a 0.7-sample arrival one tap early."""
    from uacpy.acoustic_signal import impulse_response
    fs = 8000.0
    _, h = impulse_response([1.0], [0.7 / fs], sample_rate=fs, fractional=False)
    assert np.argmax(np.abs(h)) == 1
    # fractional=True uses a windowed-sinc kernel, not a two-tap linear
    # split: the latter is a frac-dependent lowpass (-3.0 dB at
    # f/fs = 0.25 for frac = 0.5), so equal arrivals came back unequal.
    # Placed clear of the array ends, where the kernel is not truncated.
    _, hf = impulse_response([1.0], [100.7 / fs], sample_rate=fs, fractional=True,
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
        from uacpy.acoustic_signal.channel import impulse_response
        with recorded_warnings() as caught:
            _t, h = impulse_response(np.array([1.0]),
                                     np.array([position_samples / self.FS]),
                                     sample_rate=self.FS, n_samples=self.N)
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
            impulse_response([1.0], [3600.0], sample_rate=96000.0)

    @pytest.mark.parametrize("delay", [np.inf, np.nan])
    def test_nonfinite_delay_raises_typed(self, delay):
        with pytest.raises(ConfigurationError, match="default limit"):
            impulse_response([1.0], [delay], sample_rate=8000.0)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_nonpositive_or_nonfinite_sample_rate_raises(self, bad):
        with pytest.raises(ConfigurationError,
                           match="sample_rate must be > 0 Hz and finite"):
            impulse_response([1.0], [0.1], sample_rate=bad)

    def test_explicit_n_samples_is_used_as_given(self):
        t, h = impulse_response([1.0], [0.004], sample_rate=8000.0, n_samples=64)
        assert t.size == 64 and h.size == 64

    def test_short_delay_places_the_arrival(self):
        _, h = impulse_response([1.0], [0.01], sample_rate=1000.0)
        assert int(np.argmax(np.abs(h))) == 10

    def test_multipath_channel_inherits_the_tap_bound(self):
        with pytest.raises(ConfigurationError, match="default limit"):
            comms.multipath_channel([1.0], [3600.0], sample_rate=96000.0)


class TestImpulseResponseReportsBandsOutsideNyquist:
    FS = 10000.0

    def _flat(self, f0, f1):
        f = np.arange(f0, f1 + 1.0, 1.0)
        return f, np.ones(f.size, dtype=complex)

    def test_band_entirely_above_nyquist_raises_instead_of_returning_zeros(self):
        f, H = self._flat(6000.0, 6100.0)
        with pytest.raises(ConfigurationError, match="entirely above the Nyquist"):
            impulse_response_from_transfer_function(H, frequencies=f, sample_rate=self.FS)

    def test_band_straddling_nyquist_warns_and_keeps_the_part_below(self):
        f, H = self._flat(4950.0, 5050.0)
        with pytest.warns(UserWarning, match="dropped from h"):
            _t, h = impulse_response_from_transfer_function(H, frequencies=f, sample_rate=self.FS)
        f_in, H_in = self._flat(1000.0, 1100.0)
        _t, h_in = impulse_response_from_transfer_function(H_in, frequencies=f_in, sample_rate=self.FS)
        # Half the band is lost, so the peak is about half the in-band one.
        assert np.max(np.abs(h)) == pytest.approx(0.5 * np.max(np.abs(h_in)),
                                                  rel=0.05)

    def test_band_reaching_exactly_nyquist_is_silent(self):
        f = np.linspace(0.0, self.FS / 2, 65)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _t, h = impulse_response_from_transfer_function(
                np.ones(f.size, dtype=complex), frequencies=f, sample_rate=self.FS, n_samples=128)
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
        t, h = impulse_response_from_transfer_function(H, frequencies=frequencies, sample_rate=self.FS)
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
            np.ones(f.size, dtype=complex), frequencies=f, sample_rate=self.FS)
        assert h.size == int(round(self.FS / 1.0))

    def test_full_band_input_recovers_its_own_length(self):
        # The grid a full-band rfftfreq came from is returned unchanged.
        for n in (128, 129, 256):
            grid = np.fft.rfftfreq(n, 1.0 / self.FS)
            _, h = impulse_response_from_transfer_function(
                np.ones(grid.size, dtype=complex), frequencies=grid, sample_rate=self.FS)
            assert h.size == n

    def test_absurdly_fine_spacing_raises_instead_of_allocating(self):
        f = np.array([0.0, 1e-4, 2e-4])
        with pytest.raises(ConfigurationError, match="n_samples"):
            impulse_response_from_transfer_function(
                np.ones(3, dtype=complex), frequencies=f, sample_rate=1e6)
        # The limit is a default-only guard: an explicit n_samples is obeyed.
        _, h = impulse_response_from_transfer_function(
            np.ones(3, dtype=complex), frequencies=f, sample_rate=1e6, n_samples=64)
        assert h.size == 64
        assert _MAX_DEFAULT_IR_SAMPLES > 0


class TestImpulseResponseDropWarnings:
    """Every path that drops an arrival in its entirety says so: the
    quantised (``fractional=False``) path, the fractional path with an
    integer delay, and the fractional path with no kernel tap in the
    window. In-window arrivals are placed silently."""

    FS, N = 1000.0, 10

    def _run(self, delay_samples, fractional):
        with recorded_warnings() as caught:
            _t, h = impulse_response([1.0], [delay_samples / self.FS],
                                     sample_rate=self.FS, n_samples=self.N,
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
            _t, h = impulse_response([1.0], [5.0 / self.FS], sample_rate=self.FS,
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
        # 20·log10|H| floored 120 dB below its peak, the plotters' null
        # policy (no epsilon added to every value).
        magnitude = np.abs(H)
        floor = magnitude.max() * 10.0 ** (-120.0 / 20.0)
        assert np.array_equal(drawn_db,
                              20 * np.log10(np.maximum(magnitude, floor)))

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
        _t, h = impulse_response_from_transfer_function(self.H, frequencies=self.F, sample_rate=FS)
        f_back, H_back = transfer_function_from_impulse_response(
            h, FS, band=(self.F[0], self.F[-1]))
        ref = (np.interp(f_back, self.F, self.H.real)
               + 1j * np.interp(f_back, self.F, self.H.imag))
        # A scaling error shows up here and only here: the phases below
        # stay exact whatever constant multiplies H.
        assert np.abs(H_back).mean() / np.abs(ref).mean() == pytest.approx(
            1.0, abs=1e-9)
        assert np.abs(H_back - ref).max() < 1e-12

    def test_a_started_record_carries_its_offset_out(self):
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
        with pytest.raises(
                ConfigurationError,
                match='axis must be an integer|is not an axis of an array') as exc:
            transfer_function_from_impulse_response(
                self._block(), FS, axis=axis)
        assert 'axis' in str(exc.value)

    def test_one_sample_along_the_named_axis_is_refused(self):
        with pytest.raises(
                ConfigurationError,
                match='needs at least two samples along axis') as exc:
            transfer_function_from_impulse_response(
                np.zeros((2, 3, 1)), FS)
        assert 'two samples' in str(exc.value)


class TestTheBandEdgeSurvivesFloatingPointDust:
    """A band taken from the grid that made ``h`` lands ON a bin, and a
    bare ``<=`` then keeps or drops that bin depending on the last bit:
    re-inverting a sample rate moved a 995 Hz edge by 1e-13 Hz and cost a
    bin, which is a silent off-by-one in the returned spectrum."""

    def test_an_edge_bin_displaced_by_rounding_is_kept(self):
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

    def test_an_edge_genuinely_below_a_bin_excludes_it(self):
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
        with pytest.raises(ConfigurationError,
                           match=r'Field\.to_transfer_function: ') as exc:
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


class TestAnOffsetFrequencyGridIsNotInterpolated:
    """A uniform band whose first sample is not a multiple of its spacing
    sat between the DFT bins and was interpolated there, which attenuates an
    arrival at delay tau by |cos(pi tau df)|: -10.3 dB at 0.4 s on a 1 Hz
    grid starting at 100.5 Hz."""

    @pytest.mark.parametrize('f0', [100.0, 100.25, 100.5])
    def test_a_late_arrival_keeps_its_level_at_any_offset(self, f0):
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function,
        )
        f = f0 + np.arange(100) * 1.0
        H = np.exp(-2j * np.pi * f * 0.4)
        t, h = impulse_response_from_transfer_function(H, frequencies=f, sample_rate=1000.0)
        assert np.max(np.abs(h)) == pytest.approx(0.2, rel=1e-6)
        assert t[np.argmax(np.abs(h))] == pytest.approx(0.4)

    def test_an_interpolated_grid_says_so(self):
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function,
        )
        f = 100.0 + np.arange(100) * 1.0
        f[50:] += 0.5
        with pytest.warns(UserWarning, match="interpolated"):
            impulse_response_from_transfer_function(
                np.ones(100, complex), frequencies=f, sample_rate=1000.0)

    def test_a_record_of_any_length_is_the_exact_sum(self):
        # 1500 samples of a 1 s record at 1 kHz: the first 1000 are the
        # record, the rest its continuation, every one the band's sum at
        # its own instant and none interpolated.
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function,
        )
        f = 100.25 + np.arange(100) * 1.0
        H = np.exp(-2j * np.pi * f * 0.0123)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            t, h = impulse_response_from_transfer_function(
                H, frequencies=f, sample_rate=1000.0, n_samples=1500)
        direct = (2.0 * np.real(np.exp(2j * np.pi * np.outer(t, f)) @ H)
                  * (1.0 / 1000.0))
        assert h.size == 1500
        assert np.max(np.abs(h - direct)) < 1e-12 * np.max(np.abs(direct))


class TestAnOffsetGridRoundTripsExactly:
    """A grid whose first frequency is not a multiple of its spacing — most
    ``linspace`` grids, every default BROADBAND grid — is synthesised with a
    common bin offset, so its lines fall between the record's ``rfft`` bins.
    Given the grid, the way back returns ``H`` at exactly those frequencies,
    the offset rotated out and the mirror image at the negative frequencies
    removed."""

    @staticmethod
    def two_paths(f):
        return (np.exp(-2j * np.pi * f * 0.05)
                + 0.6 * np.exp(-2j * np.pi * f * 0.081 + 0.3j))

    GRIDS = {'half-bin': 100.5 + np.arange(100.0),
             'quarter-bin': 100.25 + np.arange(100.0),
             'three-tenths-bin': 25.3 + np.arange(400.0),
             'linspace': np.linspace(75.0, 125.0, 128),
             'a tenth of a bin above 0 Hz': 0.1 + np.arange(50.0),
             'a bin and a tenth below Nyquist': 449.9 + np.arange(50.0),
             # The last line lands on bin N/2 of the 1000-sample record,
             # within half a bin below the Nyquist frequency ...
             'a tenth of a bin below Nyquist': 450.9 + np.arange(50.0),
             'just under half a bin below Nyquist': 450.51 + np.arange(50.0),
             # ... or on bin N/2 - 1, just over half a bin below it.
             'just over half a bin below Nyquist': 450.49 + np.arange(50.0)}

    @pytest.mark.parametrize('name', list(GRIDS))
    def test_the_array_pair_closes_on_the_synthesis_grid(self, name):
        f = self.GRIDS[name]
        with recorded_warnings() as rec:
            _t, h = impulse_response_from_transfer_function(
                self.two_paths(f), frequencies=f, sample_rate=1000.0)
            f_back, H_back = transfer_function_from_impulse_response(
                h, 1000.0, frequencies=f)
        assert not rec, [str(w.message) for w in rec]
        np.testing.assert_array_equal(f_back, f)
        assert np.abs(H_back - self.two_paths(f)).max() < 1e-12

    @pytest.mark.parametrize('f', [np.arange(751.0, 1248.0, 4.0),
                                   np.linspace(750.0, 1250.0, 128)])
    def test_a_field_trace_comes_back_on_its_own_grid(self, f):
        from uacpy.core.results.field import Field
        field = Field(data=self.two_paths(f).reshape(1, 1, -1),
                      coords={'depth': [10.0], 'range': [100.0],
                              'frequency': f}, frequencies=f)
        back = field.to_time_trace(window=None,
                                   t_start=0.0).to_transfer_function()
        np.testing.assert_allclose(back.coords['frequency'], f, rtol=0.0,
                                   atol=1e-9)
        assert np.abs(np.ravel(back.data)
                      - self.two_paths(f)).max() < 1e-12

    def test_a_line_a_hair_off_0_hz_warns_it_cannot_be_separated(self):
        f = 1e-5 + np.arange(50.0)
        _t, h = impulse_response_from_transfer_function(
            self.two_paths(f), frequencies=f, sample_rate=1000.0)
        with pytest.warns(UserWarning, match='mirror image'):
            transfer_function_from_impulse_response(h, 1000.0, frequencies=f)

    @staticmethod
    def edge_grid(side, bins_off, n_samples, fs=1000.0):
        """50 lines spaced ``fs/n_samples``, the first ``bins_off`` of a bin
        above 0 Hz or the last ``bins_off`` of a bin below Nyquist."""
        df = fs / n_samples
        if side == '0 Hz':
            return bins_off * df + df * np.arange(50.0)
        return fs / 2.0 - bins_off * df - df * np.arange(50.0)[::-1]

    @pytest.mark.parametrize('n_samples', [1000, 1001])
    def test_a_band_a_hundredth_of_a_bin_below_nyquist_closes_quietly(
            self, n_samples):
        """The direct solve near the Nyquist frequency is as accurate as near
        0 Hz: no warning, and the accuracy it promises."""
        f = self.edge_grid('Nyquist', 0.01, n_samples)
        with recorded_warnings() as rec:
            _t, h = impulse_response_from_transfer_function(
                self.two_paths(f), frequencies=f, sample_rate=1000.0,
                n_samples=n_samples)
            _f, H_back = transfer_function_from_impulse_response(
                h, 1000.0, frequencies=f)
        assert not rec, [str(w.message) for w in rec]
        assert np.abs(H_back - self.two_paths(f)).max() < 1e-9

    @pytest.mark.parametrize('n_samples', [1000, 1001])
    @pytest.mark.parametrize('side', ['0 Hz', 'Nyquist'])
    @pytest.mark.parametrize('bins_off', [1e-5, 1e-8])
    def test_a_line_a_hair_off_an_edge_is_as_accurate_as_its_warning_states(
            self, bins_off, side, n_samples):
        """A line closer to its mirror image leaves an ill-conditioned
        solve; the warning states the accuracy it has, above one when no
        digit is left (1e-8 of a bin), and the error stays within it."""
        f = self.edge_grid(side, bins_off, n_samples)
        H = self.two_paths(f)
        _t, h = impulse_response_from_transfer_function(
            H, frequencies=f, sample_rate=1000.0, n_samples=n_samples)
        with pytest.warns(NumericsWarning, match='mirror image') as rec:
            _f, H_back = transfer_function_from_impulse_response(
                h, 1000.0, frequencies=f)
        stated = float(re.search(r'accurate to about (\S+) of its peak',
                                 str(rec[0].message)).group(1))
        assert np.abs(H_back - H).max() / np.abs(H).max() <= stated
        assert (stated > 1.0) == (bins_off < 1e-6)

    def test_an_odd_record_is_refused_a_line_on_its_nyquist_frequency(self):
        """An odd record's Nyquist frequency falls half a bin off its
        lattice; a line there is its own mirror image, whose sine the record
        never samples. A grid 1e-8 of a bin short of it is answered (above)."""
        f = self.edge_grid('Nyquist', 0.0, 1001)
        assert f[-1] == 500.0
        with pytest.raises(ConfigurationError,
                           match='^transfer_function_from_impulse_response: '
                                 '.*Nyquist frequency 500 Hz'):
            transfer_function_from_impulse_response(np.zeros(1001), 1000.0,
                                                    frequencies=f)

    @pytest.mark.parametrize('n_samples', [1000, 1001])
    def test_h_on_the_nyquist_frequency_loses_its_sine_with_a_warning(
            self, n_samples):
        """A real response holds only the cosine of a line on fs/2: the
        forward says it drops an imaginary part above 1e-9 of the peak, and
        the even record's way back returns the real part."""
        f = self.edge_grid('Nyquist', 0.0, n_samples)
        H = self.two_paths(f)
        with pytest.warns(NumericsWarning,
                          match='Nyquist frequency 500 Hz has an imaginary'):
            _t, h = impulse_response_from_transfer_function(
                H, frequencies=f, sample_rate=1000.0, n_samples=n_samples)
        if n_samples % 2 == 0:
            _f, H_back = transfer_function_from_impulse_response(
                h, 1000.0, frequencies=f)
            assert abs(H_back[-1] - H[-1].real) < 1e-12
            assert np.abs(H_back[:-1] - H[:-1]).max() < 1e-12

    @pytest.mark.parametrize('n_samples', [1000, 1001])
    @pytest.mark.parametrize('imag, warns', [(1e-10, False), (1e-8, True)])
    def test_the_nyquist_warning_ignores_rounding(self, n_samples, imag,
                                                  warns):
        f = self.edge_grid('Nyquist', 0.0, n_samples)
        H = np.ones(f.size, dtype=complex)
        H[-1] += 1j * imag
        with recorded_warnings() as rec:
            impulse_response_from_transfer_function(
                H, frequencies=f, sample_rate=1000.0, n_samples=n_samples)
        assert any('Nyquist frequency 500 Hz has an imaginary'
                   in str(w.message) for w in rec) == warns

    @pytest.mark.parametrize('kwargs, match', [
        (dict(frequencies=100.5 + np.arange(10.0), band=(100.0, 110.0)),
         'not both'),
        (dict(frequencies=100.5 + 2.0 * np.arange(10.0)), 'spaced'),
        (dict(frequencies=495.5 + np.arange(10.0)), 'holds 0 to 500'),
    ])
    def test_a_grid_the_record_cannot_hold_is_refused(self, kwargs, match):
        with pytest.raises(ConfigurationError,
                           match='^transfer_function_from_impulse_response: '
                                 f'.*{match}'):
            transfer_function_from_impulse_response(np.zeros(1000), 1000.0,
                                                    **kwargs)


class TestAnArrivalGridSharesOneClock:
    """``simulate_arrival_grid`` synthesises every receiver of a grid from its
    own arrival list on ONE window spanning every cell's arrivals."""

    FS = 8000.0
    FC = 1000.0

    def _pulse(self):
        t = np.arange(64) / self.FS
        return np.sin(2 * np.pi * self.FC * t) * np.hanning(t.size)

    @staticmethod
    def _cell(delays, amplitudes=None):
        delays = np.asarray(delays, dtype=float)
        amps = (np.ones_like(delays) if amplitudes is None
                else np.asarray(amplitudes, dtype=float))
        return {'n_arrivals': delays.size, 'amplitudes': amps,
                'delays': delays, 'delays_imag': np.zeros_like(delays),
                'phases': np.zeros_like(delays)}

    def test_every_cell_is_its_own_reception_on_the_shared_window(self):
        from uacpy.acoustic_signal import (simulate_arrival_grid,
                                           simulate_arrival_reception)
        cells = [[self._cell([0.10]), self._cell([0.30, 0.31], [1.0, 0.5])]]
        t, traces = simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                          sample_rate=self.FS,
                                          trace_frequency=self.FC)
        assert traces.shape == (1, 2, t.size)
        for j, cell in enumerate(cells[0]):
            _, ref = simulate_arrival_reception(cell['amplitudes'],
                                                cell['delays'],
                                                source_waveform=self._pulse(),
                                                sample_rate=self.FS,
                                                trace_frequency=self.FC,
                                                output_duration=t[-1] - t[0] + 1.0 / self.FS,
                                                t_start=t[0], report={})
            np.testing.assert_array_equal(traces[0, j], ref[:t.size])

    def test_the_window_spans_the_earliest_and_latest_arrival_of_the_grid(
            self):
        from uacpy.acoustic_signal import simulate_arrival_grid
        cells = [[self._cell([0.10])], [self._cell([0.50])]]
        t, traces = simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                          sample_rate=self.FS,
                                          trace_frequency=self.FC)
        assert t[0] <= 0.10 and t[-1] >= 0.50
        assert np.max(np.abs(traces[1, 0])) > 0.0

    def test_a_cell_no_arrival_reached_is_silent(self):
        from uacpy.acoustic_signal import simulate_arrival_grid
        cells = [[self._cell([0.10]), self._cell([])]]
        _, traces = simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                          sample_rate=self.FS,
                                          trace_frequency=self.FC)
        assert np.all(traces[0, 1] == 0.0)
        assert np.any(traces[0, 0] != 0.0)

    def test_an_empty_list_is_silent_alone_and_on_a_grid(self):
        """An empty arrival list is the empty sum, alone and as a cell of a
        grid: zeros, not NaN."""
        from uacpy.acoustic_signal import (
            arrival_grid_impulse_response, arrival_grid_transfer_function,
            arrival_transfer_function, delayandsum, simulate_arrival_grid,
            simulate_arrival_reception,
        )
        freqs = np.linspace(900.0, 1100.0, 5)
        _, alone = simulate_arrival_reception([], [],
                                              source_waveform=self._pulse(),
                                              sample_rate=self.FS,
                                              trace_frequency=self.FC)
        rts, _ = delayandsum({'n_arrivals': 0}, source_waveform=self._pulse(),
                             sample_rate=self.FS, trace_frequency=self.FC)
        H_alone = arrival_transfer_function(freqs, [], [])
        assert alone.size and np.all(alone == 0.0)
        assert rts.size and np.all(rts == 0.0)
        assert H_alone.shape == (5,) and np.all(H_alone == 0.0)
        cells = [[self._cell([0.10]), self._cell([])]]
        _, traces = simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                          sample_rate=self.FS,
                                          trace_frequency=self.FC)
        H_grid = arrival_grid_transfer_function(freqs, cells)
        _, h_grid = arrival_grid_impulse_response(cells, sample_rate=self.FS)
        assert np.all(traces[0, 1] == 0.0)
        assert np.all(H_grid[0, 1] == 0.0)
        assert np.all(h_grid[0, 1] == 0.0)

    def test_echoes_outside_a_pinned_window_are_named_once(self):
        from uacpy.acoustic_signal.channel import _simulate_arrival_grid
        cells = [[self._cell([0.10, 0.90]), self._cell([0.12, 0.95])]]
        with recorded_warnings() as rec:
            _simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                   sample_rate=self.FS,
                                   trace_frequency=self.FC, t_start=0.0,
                                   output_duration=0.5, who='probe')
        named = [w for w in rec if str(w.message).startswith('probe')]
        assert len(named) == 1

    @staticmethod
    def _lossy_cell(delays, imag):
        delays = np.asarray(delays, dtype=float)
        return {'n_arrivals': delays.size, 'amplitudes': np.ones_like(delays),
                'delays': delays, 'delays_imag': np.asarray(imag, float),
                'phases': np.zeros_like(delays)}

    def test_a_cell_does_not_depend_on_the_cells_beside_it(self):
        # Each cell is its own reception on the shared window: the same
        # cell beside a far longer one comes back identical.
        from uacpy.acoustic_signal import simulate_arrival_grid
        from uacpy.core.absorption import Thorp
        near = self._lossy_cell([0.10, 0.12], [-1e-5, 0.0])
        alone = [[near, self._cell([0.105])]]
        beside = [[near, self._cell([0.105, 0.40])]]
        kw = dict(source_waveform=self._pulse(), sample_rate=self.FS,
                  trace_frequency=self.FC, absorption=Thorp(),
                  t_start=0.05, output_duration=0.5)
        _, a = simulate_arrival_grid(alone, **kw)
        _, b = simulate_arrival_grid(beside, **kw)
        np.testing.assert_array_equal(a[0, 0], b[0, 0])

    def test_a_lossy_cell_on_the_grid_is_its_own_reception(self):
        from uacpy.acoustic_signal import (simulate_arrival_grid,
                                           simulate_arrival_reception)
        from uacpy.core.absorption import Thorp
        cells = [[self._lossy_cell([0.10, 0.12], [-1e-5, 0.0]),
                  self._cell([0.30])]]
        t, traces = simulate_arrival_grid(cells, source_waveform=self._pulse(),
                                          sample_rate=self.FS,
                                          trace_frequency=self.FC,
                                          absorption=Thorp())
        for j, cell in enumerate(cells[0]):
            _, ref = simulate_arrival_reception(cell['amplitudes'],
                                                cell['delays'],
                                                source_waveform=self._pulse(),
                                                sample_rate=self.FS,
                                                trace_frequency=self.FC,
                                                delays_imag_s=cell['delays_imag'],
                                                output_duration=t.size / self.FS,
                                                t_start=t[0], report={},
                                                absorption=Thorp())
            np.testing.assert_allclose(traces[0, j], ref, rtol=0.0,
                                       atol=1e-12 * np.abs(ref).max())


class TestTheReceptionWindowEndsAfterTheLastCopy:
    """Left unset, a reception's window runs to the latest arrival plus one
    source duration — where the last copy of the waveform ends — plus a quiet
    tail of one source duration, capped at the arrivals' delay span and
    floored at the fractional-delay kernel's ring."""

    FS = 8000.0

    def _length(self, n_source, delays):
        from uacpy.acoustic_signal import simulate_arrival_reception
        with recorded_warnings() as rec:
            t, rts = simulate_arrival_reception(np.ones(len(delays)), delays,
                                                source_waveform=np.hanning(n_source),
                                                sample_rate=self.FS,
                                                trace_frequency=1000.0)
        assert not rec, [str(w.message) for w in rec]
        assert t.size == rts.size
        return t.size, float(t[0])

    def test_a_long_signal_gets_a_tail_of_the_delay_span(self):
        n, t0 = self._length(8000, [0.50, 0.52])      # 1 s signal, 20 ms span
        assert n == int(np.ceil(((0.52 - t0) + 1.0 + 0.02) * self.FS))

    def test_a_pulse_shorter_than_the_span_gets_a_tail_of_one_pulse(self):
        n, t0 = self._length(64, [0.10, 0.30])        # 8 ms pulse, 200 ms span
        assert n == int(np.ceil(((0.30 - t0) + 2 * 0.008) * self.FS))

    def test_one_arrival_gets_the_kernel_ring_as_its_tail(self):
        from uacpy.acoustic_signal.channel import _FRAC_DELAY_HALF_LEN
        n, t0 = self._length(64, [0.10])
        assert n == int(np.ceil(((0.10 - t0) + 0.008 + _FRAC_DELAY_HALF_LEN
                                 / self.FS) * self.FS))


class TestASignalIsPlayedThroughTapsByLinearConvolution:
    """``apply_impulse_response`` is the linear convolution of a waveform
    with sampled taps — ``N + M − 1`` samples from the time of ``h[0]`` —
    by FFT overlap-add, and the direct sum below 19 taps (Abraham sect.
    4.9.4)."""

    @staticmethod
    def _two_path_taps():
        # Two arrivals, the second inverted and between samples.
        _, h = impulse_response([1.0, -0.5], [0.0101, 0.0347], sample_rate=FS)
        return h

    def test_a_two_path_pulse_is_the_direct_convolution(self):
        from uacpy.acoustic_signal import apply_impulse_response, tone_burst
        h = self._two_path_taps()
        x = tone_burst(1000.0, 10, sample_rate=FS)[1]
        t, y = apply_impulse_response(h, source_waveform=x, sample_rate=FS,
                                      t_start=0.25)
        want = np.convolve(x, h)
        assert y.size == x.size + h.size - 1
        assert t[0] == 0.25 and t[1] - t[0] == pytest.approx(1.0 / FS)
        assert np.max(np.abs(y - want)) < 1e-12 * np.max(np.abs(want))

    def test_every_cell_of_a_block_of_responses_is_applied_at_once(self):
        from uacpy.acoustic_signal import apply_impulse_response
        rng = np.random.default_rng(0)
        h = rng.standard_normal((2, 3, 400))
        x = rng.standard_normal(3000)
        _, y = apply_impulse_response(np.moveaxis(h, -1, 1),
                                      source_waveform=x, sample_rate=FS,
                                      axis=1)
        assert y.shape == (2, 3399, 3)
        for i in range(2):
            for j in range(3):
                want = np.convolve(x, h[i, j])
                assert np.max(np.abs(y[i, :, j] - want)) < 1e-12 * np.max(
                    np.abs(want))

    def test_a_signal_longer_than_the_record_does_not_fold(self):
        # 3 s of noise through the taps of a 2 s record: the convolution is
        # 5 s long and linear, where a Fourier synthesis on the same grid
        # is refused.
        from uacpy.acoustic_signal import (apply_impulse_response,
                                           synthesize_time_series)
        rng = np.random.default_rng(1)
        fs = 1000.0
        f = np.arange(100.0, 300.0, 0.5)
        H = np.exp(-2j * np.pi * f * 0.10) + 0.6 * np.exp(-2j * np.pi * f * 0.35)
        _, h = impulse_response_from_transfer_function(H, frequencies=f,
                                                       sample_rate=fs)
        x = rng.standard_normal(3000)
        _, y = apply_impulse_response(h, source_waveform=x, sample_rate=fs)
        want = np.convolve(x, h)
        assert y.size == 3000 + 2000 - 1
        assert np.max(np.abs(y - want)) < 1e-12 * np.max(np.abs(want))
        with pytest.raises(ConfigurationError, match="folded onto itself"):
            synthesize_time_series(H, frequencies=f, source_waveform=x,
                                   sample_rate=fs, t_start=0.0)

    @pytest.mark.parametrize('n_taps, exact', [(18, True), (19, False)])
    def test_below_nineteen_taps_the_sum_is_direct(self, n_taps, exact):
        from uacpy.acoustic_signal import apply_impulse_response
        rng = np.random.default_rng(2)
        h = rng.standard_normal(n_taps)
        x = rng.standard_normal(5000)
        _, y = apply_impulse_response(h, source_waveform=x, sample_rate=FS)
        want = np.convolve(x, h)
        assert np.array_equal(y, want) is exact
        assert np.max(np.abs(y - want)) < 1e-12 * np.max(np.abs(want))

    @pytest.mark.parametrize('h, x, t_start, match', [
        ([], np.ones(4), 0.0, 'no taps'),
        ([1.0], np.ones((2, 4)), 0.0, 'non-empty 1-D'),
        ([1.0], np.array([1.0, np.nan]), 0.0, 'finite numbers'),
        ([1.0], np.ones(4), np.inf, 't_start must be one finite time'),
    ])
    def test_what_it_cannot_convolve_is_refused(self, h, x, t_start, match):
        from uacpy.acoustic_signal import apply_impulse_response
        with pytest.raises(ConfigurationError,
                           match=f"apply_impulse_response: .*{match}"):
            apply_impulse_response(h, source_waveform=x, sample_rate=FS,
                                   t_start=t_start)


class TestAStreamedSignalIsTheWholeConvolution:
    """``apply_impulse_response_blocks``: overlap-add over a signal that
    arrives in blocks, in memory set by the block and the response, not by
    the signal."""

    @staticmethod
    def _stream(x, sizes):
        start = 0
        for size in sizes:
            yield x[start:start + size]
            start += size
        if start < x.size:
            yield x[start:]

    @pytest.mark.parametrize('block_size, sizes', [
        (None, [700] * 10),
        (1000, [333] * 30),
        (100, [5000, 1, 2, 3]),          # a segment shorter than the response
        (4096, [10000]),
    ])
    def test_the_blocks_concatenate_to_the_in_memory_result(self, block_size,
                                                            sizes):
        from uacpy.acoustic_signal import (apply_impulse_response,
                                           apply_impulse_response_blocks)
        rng = np.random.default_rng(3)
        h = rng.standard_normal(700)
        x = rng.standard_normal(10000)
        _, want = apply_impulse_response(h, source_waveform=x, sample_rate=FS)
        got = np.concatenate(list(apply_impulse_response_blocks(
            h, self._stream(x, sizes), block_size=block_size)))
        assert got.size == want.size
        assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))

    def test_a_complex_signal_and_a_block_of_responses_stream(self):
        from uacpy.acoustic_signal import (apply_impulse_response,
                                           apply_impulse_response_blocks)
        rng = np.random.default_rng(4)
        h = rng.standard_normal((3, 50))
        x = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
        _, want = apply_impulse_response(h, source_waveform=x, sample_rate=FS)
        got = np.concatenate(list(apply_impulse_response_blocks(
            h, self._stream(x, [512] * 4), block_size=256)), axis=-1)
        assert np.iscomplexobj(got) and got.shape == want.shape
        assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))

    def test_memory_does_not_grow_with_the_signal(self):
        # 10 s and 40 s of 48 kHz noise through a 0.5 s channel (24 000
        # taps; transform blocks of 168 001 samples, so both runs stream
        # whole blocks), generated block by block: the in-memory form holds
        # the whole signal and output, the streaming form one transform
        # block.
        import tracemalloc
        from uacpy.acoustic_signal import apply_impulse_response_blocks
        fs = 48000.0
        h = np.random.default_rng(5).standard_normal(24000) * 1e-3

        def peak(seconds):
            rng = np.random.default_rng(6)
            blocks = (rng.standard_normal(4800)
                      for _ in range(int(seconds * fs) // 4800))
            tracemalloc.start()
            total = 0
            for out in apply_impulse_response_blocks(h, blocks):
                total += out.size
            used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert total == int(seconds * fs) + h.size - 1
            return used

        short, long = peak(10.0), peak(40.0)
        assert long < 1.2 * short
        assert long < 0.5 * (8 * 2 * 40 * fs)   # under half of x and y

    def test_a_stream_with_no_samples_is_refused(self):
        from uacpy.acoustic_signal import apply_impulse_response_blocks
        with pytest.raises(ConfigurationError, match="no samples"):
            list(apply_impulse_response_blocks([1.0, 0.5], iter([])))

    def test_a_block_that_is_not_a_signal_is_refused(self):
        from uacpy.acoustic_signal import apply_impulse_response_blocks
        with pytest.raises(ConfigurationError, match="block 1 must be"):
            list(apply_impulse_response_blocks(
                [1.0, 0.5], [np.ones(4), np.ones((2, 2))]))


class TestTheResponseFromHHasItsOwnStart:
    """``impulse_response_from_transfer_function(t_start=)`` places the
    record off the bulk delay, one start per cell when the cells differ,
    every sample the band's exact sum at its own instant."""

    def test_the_record_opens_at_t_start_not_at_the_emission(self):
        # An arrival at 3.333 s on a 2 Hz grid (0.5 s record): from 0 it
        # folds to 0.333 s, from t_start = 3.3 s it is where it is.
        f = np.arange(100.0, 200.0, 2.0)
        H = np.exp(-2j * np.pi * f * 3.333)
        t, h = impulse_response_from_transfer_function(
            H, frequencies=f, sample_rate=1000.0, t_start=3.3)
        assert t[0] == pytest.approx(3.3) and h.size == 500
        assert t[np.argmax(np.abs(h))] == pytest.approx(3.333, abs=1e-3)

    def test_each_cell_has_its_own_start(self):
        f = np.arange(0.0, 500.0, 2.0)
        delays = np.array([[0.2, 1.3], [2.45, 7.7]])
        H = np.exp(-2j * np.pi * f[None, None, :] * delays[..., None])
        starts = delays - 0.1
        t, h = impulse_response_from_transfer_function(
            H, frequencies=f, sample_rate=1000.0, t_start=starts)
        assert t.shape == (2, 2, 500) and h.shape == (2, 2, 500)
        peaks = np.take_along_axis(t, np.argmax(np.abs(h), axis=-1)[..., None],
                                   axis=-1)[..., 0]
        np.testing.assert_allclose(peaks, delays, atol=1e-9)

    @pytest.mark.parametrize('fs', [1000.0, 1001.3])
    def test_a_uniform_grid_is_summed_exactly_at_any_rate(self, fs):
        # 1000 Hz is the 2 Hz grid's own lattice (an inverse FFT); 1001.3 Hz
        # is not (chirp-z): both are the band's sum at t_start + n/fs.
        f = 100.3 + np.arange(150) * 2.0
        H = np.exp(-2j * np.pi * f * 0.1234) * (1.0 + 0.3j)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            t, h = impulse_response_from_transfer_function(
                H, frequencies=f, sample_rate=fs, t_start=0.05)
        direct = 2.0 * np.real(np.exp(2j * np.pi * np.outer(t, f)) @ H) * (
            2.0 / fs)
        assert h.size == int(np.ceil(fs / 2.0 - 1e-9))
        assert np.max(np.abs(h - direct)) < 1e-11 * np.max(np.abs(direct))

    def test_a_long_chirp_z_record_keeps_its_phase(self):
        # 350 366 samples at 48 kHz on a 0.137 Hz grid (no whole fs/Δf): the
        # chirp phase pi q k^2 runs to 1.1e6 rad, and a chirp powered from
        # the complex step exp(i pi q) loses 1e-7 of the peak; checked at
        # the record's start, middle and end against the sum with every
        # phase reduced in whole cycles.
        fs, df = 48000.0, 0.137
        f = 1000.0 + df * np.arange(20000)
        H = np.exp(1j * np.random.default_rng(8).uniform(-np.pi, np.pi,
                                                         f.size))
        t, h = impulse_response_from_transfer_function(H, frequencies=f,
                                                       sample_rate=fs)
        assert h.size == int(np.ceil(fs / df))
        k = np.arange(f.size, dtype=np.int64)
        picks = [0, 1, h.size // 2, h.size - 1]
        direct = []
        for m in picks:
            turns = np.remainder(np.remainder(m * 1000.0 / fs, 1.0)
                                 + np.remainder((m * k) * (df / fs), 1.0), 1.0)
            direct.append(2.0 * np.real(np.sum(H * np.exp(2j * np.pi * turns)))
                          * df / fs)
        direct = np.array(direct)
        assert np.max(np.abs(h[picks] - direct)) < 1e-9 * np.max(np.abs(h))

    def test_a_start_per_cell_must_broadcast_against_the_cells(self):
        f = np.arange(0.0, 500.0, 2.0)
        with pytest.raises(ConfigurationError, match="does not broadcast"):
            impulse_response_from_transfer_function(
                np.ones((2, 3, f.size), complex), frequencies=f,
                sample_rate=1000.0, t_start=np.zeros(2))


class TestAnArrivalReceptionIsTheExactProduct:
    """``simulate_arrival_reception`` is the waveform and its Hilbert
    transform (on a record padded four waveform lengths either side) through
    the arrival list's in-phase and quadrature taps, which is the exact
    product ``X·H`` (on a 2**20-sample record): measured 1e-8 of the peak
    for a 10 kHz pulse through 20 lossy paths over 50 ms and 1.7e-10 for a
    lone quarter-turn arrival between samples. The phase rotation is the
    waveform's, so a pulse whose content lies below one over the taps'
    margin (188 Hz at 256 samples at 48 kHz) is rotated as exactly: a
    4-cycle 50 Hz pulse 1.7e-5 through a lone quarter turn and 3.3e-5
    through two paths 5 ms apart, where taps rotating the phase themselves
    were 0.55 and 0.67 of the peak off, and the delay-and-sum before them
    (a Hilbert transform on the waveform's own length) 2.5e-3 and 4.9e-3."""

    FS, FC = 48000.0, 10000.0

    def _exact(self, x, amps, delays, t0, **kw):
        from uacpy.acoustic_signal import arrival_transfer_function
        L = 1 << 20
        f = np.fft.rfftfreq(L, 1.0 / self.FS)
        H = arrival_transfer_function(f, amps, np.asarray(delays) - t0, **kw)
        H[0], H[-1] = H[0].real, H[-1].real
        return np.fft.irfft(np.fft.rfft(x, L) * H, L)

    def _pulse(self):
        n = 960
        return (np.sin(2 * np.pi * self.FC * np.arange(n) / self.FS)
                * np.hanning(n))

    def test_twenty_lossy_paths(self):
        from uacpy.acoustic_signal import simulate_arrival_reception
        from uacpy.core.absorption import Thorp
        rng = np.random.default_rng(7)
        d = np.sort(1.3 + rng.uniform(0.0, 0.05, 20))
        amp = rng.uniform(0.05, 1.0, 20)
        ph = rng.uniform(-np.pi, np.pi, 20)
        di = -rng.uniform(0.0, 2e-6, 20)
        x = self._pulse()
        t, y = simulate_arrival_reception(
            amp, d, source_waveform=x, sample_rate=self.FS,
            trace_frequency=self.FC, delays_imag_s=di, phases_rad=ph,
            absorption=Thorp())
        exact = self._exact(x, amp, d, t[0], delays_imag_s=di,
                            phases_rad=ph, trace_frequency=self.FC,
                            absorption=Thorp())[:y.size]
        assert np.max(np.abs(y - exact)) < 1e-7 * np.max(np.abs(exact))

    def test_twenty_lossy_paths_through_the_impulse_response(self):
        # The waveform played through arrival_impulse_response's taps: each
        # path's kernel under its Hann window, within 1e-6 of the peak
        # (measured 2.1e-8; cut square, the kernels miss by 2.1e-5).
        from uacpy.acoustic_signal import (apply_impulse_response,
                                           arrival_impulse_response)
        from uacpy.core.absorption import Thorp
        rng = np.random.default_rng(7)
        d = np.sort(1.3 + rng.uniform(0.0, 0.05, 20))
        amp = rng.uniform(0.05, 1.0, 20)
        ph = rng.uniform(-np.pi, np.pi, 20)
        di = -rng.uniform(0.0, 2e-6, 20)
        x = self._pulse()
        kw = dict(delays_imag_s=di, phases_rad=ph, trace_frequency=self.FC,
                  absorption=Thorp())
        th, h = arrival_impulse_response(amp, d, sample_rate=self.FS, **kw)
        t, y = apply_impulse_response(h, source_waveform=x,
                                      sample_rate=self.FS, t_start=th[0])
        exact = self._exact(x, amp, d, t[0], **kw)[:y.size]
        assert np.max(np.abs(y - exact)) < 1e-6 * np.max(np.abs(exact))

    def test_a_lone_quarter_turn_between_samples(self):
        from uacpy.acoustic_signal import simulate_arrival_reception
        x = self._pulse()
        t, y = simulate_arrival_reception(
            [1.0], [0.0123456], source_waveform=x, sample_rate=self.FS,
            trace_frequency=self.FC, phases_rad=[np.pi / 2])
        exact = self._exact(x, [1.0], [0.0123456], t[0],
                            phases_rad=[np.pi / 2])[:y.size]
        assert np.max(np.abs(y - exact)) < 1e-7 * np.max(np.abs(exact))

    def test_two_hundred_lossy_paths_over_a_second(self):
        # Each path's kernel under its Hann window: within 1e-5 of the peak
        # (cut square, the kernels miss by about 4e-5).
        from uacpy.acoustic_signal import simulate_arrival_reception
        from uacpy.core.absorption import Thorp
        rng = np.random.default_rng(11)
        d = np.sort(0.5 + rng.uniform(0.0, 1.0, 200))
        amp = rng.uniform(0.05, 1.0, 200)
        ph = rng.uniform(-np.pi, np.pi, 200)
        di = -rng.uniform(0.0, 2e-5, 200)
        n = 960
        x = (np.sin(2 * np.pi * 3000.0 * np.arange(n) / self.FS)
             * np.hanning(n))
        t, y = simulate_arrival_reception(
            amp, d, source_waveform=x, sample_rate=self.FS,
            trace_frequency=3000.0, delays_imag_s=di, phases_rad=ph,
            absorption=Thorp())
        exact = self._exact(x, amp, d, t[0], delays_imag_s=di,
                            phases_rad=ph, trace_frequency=3000.0,
                            absorption=Thorp())[:y.size]
        assert np.max(np.abs(y - exact)) < 1e-5 * np.max(np.abs(exact))

    def test_the_omitted_share_is_weighed_with_its_absorption(self):
        # Equal magnitudes; the echo past the window is absorbed by
        # exp(2 pi 1000 Hz * -2e-4 s): -11 dB of the received energy, where
        # the magnitudes alone would say -3 dB.
        from uacpy.acoustic_signal import simulate_arrival_reception
        x = np.hanning(64)
        with recorded_warnings() as record:
            simulate_arrival_reception(
                [1.0, 1.0], [0.05, 0.15], source_waveform=x,
                sample_rate=self.FS, trace_frequency=1000.0,
                delays_imag_s=[0.0, -2e-4], t_start=0.0,
                output_duration=0.1)
        said = [str(w.message) for w in record
                if 'fall entirely outside' in str(w.message)]
        assert len(said) == 1, said
        assert '-11 dB of the received energy' in said[0]

    def test_an_echo_off_the_window_is_left_out_whole(self):
        # The quarter-turn echo at 0.11 s starts after the [0, 0.1) s window
        # and is reported omitted; its Hilbert transform's precursor would
        # still reach in, so it is left out of the reception whole: the
        # window holds the echo at 0.05 s alone, sample for sample.
        from uacpy.acoustic_signal import simulate_arrival_reception
        kw = dict(source_waveform=self._pulse(), sample_rate=self.FS,
                  trace_frequency=self.FC, t_start=0.0, output_duration=0.1)
        with recorded_warnings() as record:
            _, both = simulate_arrival_reception(
                [1.0, 1.0], [0.05, 0.11], phases_rad=[0.0, np.pi / 2], **kw)
        assert any('1 echo(es) fall entirely outside' in str(w.message)
                   for w in record)
        _, alone = simulate_arrival_reception([1.0], [0.05],
                                              phases_rad=[0.0], **kw)
        np.testing.assert_array_equal(both, alone)

    def _burst(self, f0, n_cycles):
        n = int(round(n_cycles / f0 * self.FS))
        return np.sin(2 * np.pi * f0 * np.arange(n) / self.FS) * np.hanning(n)

    @pytest.mark.parametrize('amps, delays, phases', [
        ([1.0], [0.0123456], [np.pi / 2]),
        ([1.0, 0.7], [0.1, 0.105], [0.0, -np.pi / 2]),
    ])
    def test_a_pulse_below_the_taps_reach_is_rotated_exactly(
            self, amps, delays, phases):
        # A 4-cycle 50 Hz pulse at 48 kHz lies below one over a 256-sample
        # margin: within 1e-4 of the peak (measured 1.7e-5 and 3.3e-5).
        from uacpy.acoustic_signal import simulate_arrival_reception
        x = self._burst(50.0, 4)
        t, y = simulate_arrival_reception(
            amps, delays, source_waveform=x, sample_rate=self.FS,
            trace_frequency=50.0, phases_rad=phases)
        exact = self._exact(x, amps, delays, t[0],
                            phases_rad=phases)[:y.size]
        assert np.max(np.abs(y - exact)) < 1e-4 * np.max(np.abs(exact))


class TestAStartTimeIsOneNumberOfSeconds:
    """The plain functions place their output at a stated time: ``t_start``
    is one finite number of seconds (one per cell where a function takes a
    start per cell). The estimated placements ``None`` and ``'reduced'`` are
    a Field method's, and are refused here by name with that remedy, not
    left to ``float()``'s TypeError or ValueError."""

    F = np.linspace(100.0, 300.0, 21)
    WF = np.hanning(50)

    def _calls(self):
        from uacpy.acoustic_signal import (
            apply_impulse_response, arrival_grid_impulse_response,
            arrival_impulse_response, simulate_arrival_reception,
            synthesize_time_series)
        H = np.ones(self.F.size, dtype=complex)
        return {
            'synthesize_time_series': lambda t: synthesize_time_series(
                H, frequencies=self.F, source_waveform=self.WF,
                sample_rate=8000.0, t_start=t),
            'apply_impulse_response': lambda t: apply_impulse_response(
                np.ones(5), source_waveform=self.WF, sample_rate=8000.0,
                t_start=t),
            'simulate_arrival_reception':
                lambda t: simulate_arrival_reception(
                    [1.0], [0.1], source_waveform=self.WF, sample_rate=8000.0,
                    trace_frequency=200.0, t_start=t),
            'impulse_response_from_transfer_function':
                lambda t: impulse_response_from_transfer_function(
                    H, frequencies=self.F, sample_rate=8000.0, t_start=t),
            'arrival_impulse_response': lambda t: arrival_impulse_response(
                [1.0], [0.1], sample_rate=8000.0, t_start=t),
            'arrival_grid_impulse_response':
                lambda t: arrival_grid_impulse_response(
                    [[{'amplitudes': [1.0], 'delays': [0.1]}]],
                    sample_rate=8000.0, t_start=t),
        }

    @pytest.mark.parametrize('name', [
        'synthesize_time_series', 'apply_impulse_response',
        'simulate_arrival_reception',
        'impulse_response_from_transfer_function',
        'arrival_impulse_response', 'arrival_grid_impulse_response'])
    @pytest.mark.parametrize('t_start', [0.05, np.float64(0.05),
                                         np.array(0.05)])
    def test_a_number_of_seconds_is_taken(self, name, t_start):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t, _ = self._calls()[name](t_start)
        assert np.ravel(t)[0] == pytest.approx(0.05)

    @pytest.mark.parametrize('name', [
        'synthesize_time_series', 'apply_impulse_response',
        'simulate_arrival_reception', 'arrival_impulse_response'])
    @pytest.mark.parametrize('t_start', [None, 'reduced', [0.0, 1.0],
                                         np.nan, True, np.array(True),
                                         np.complex128(0.05 + 1j), '0.05'])
    def test_anything_else_is_refused_by_the_call(self, name, t_start):
        if (name in ('simulate_arrival_reception', 'arrival_impulse_response')
                and t_start is None):
            pytest.skip("None is the function's own estimate")
        with pytest.raises(ConfigurationError,
                           match=f"^{name}: t_start must be one finite "
                                 f"time in seconds"):
            self._calls()[name](t_start)

    @pytest.mark.parametrize('name', [
        'impulse_response_from_transfer_function',
        'arrival_grid_impulse_response'])
    @pytest.mark.parametrize('t_start', ['reduced', np.array([True]),
                                         np.array([0.05 + 1j])])
    def test_only_real_times_are_taken_where_a_start_per_cell_is(
            self, name, t_start):
        with pytest.raises(ConfigurationError,
                           match=f"^{name}: t_start must be finite times"):
            self._calls()[name](t_start)


class TestAPathOnASampleIsOneTap:
    """A path on a sample, at phase 0, is one tap: white noise through it
    comes back as the noise itself, delayed, to rounding (1e-10 of it;
    a dropped Nyquist line costs 1/N of the record), whatever the
    parity of the record — each path's impulse is built on an odd record of
    its own, which has no Nyquist line to drop, and an impulse response
    keeps the Nyquist line's cosine on an even record."""

    FS = 48000.0
    NOISE = np.random.default_rng(1).standard_normal(4800)

    @pytest.mark.parametrize('t_start', [0.15, 0.2, 0.2 - 1.0 / 48000.0])
    def test_a_reception_is_the_delayed_noise(self, t_start):
        from uacpy.acoustic_signal import simulate_arrival_reception
        t, y = simulate_arrival_reception(
            [1.0], [0.25], source_waveform=self.NOISE, sample_rate=self.FS,
            trace_frequency=1000.0, t_start=t_start)
        at = int(round((0.25 - t[0]) * self.FS))
        want = np.zeros(y.size)
        want[at:at + self.NOISE.size] = self.NOISE[:y.size - at]
        assert np.max(np.abs(y - want)) < 1e-10 * np.max(np.abs(self.NOISE))

    @pytest.mark.parametrize('n_samples', [1024, 1025])
    def test_an_impulse_response_is_one_tap(self, n_samples):
        from uacpy.acoustic_signal import arrival_impulse_response
        t, h = arrival_impulse_response([1.0], [0.25], sample_rate=self.FS,
                                        t_start=0.25 - 480 / self.FS,
                                        n_samples=n_samples)
        want = np.zeros(n_samples)
        want[480] = 1.0
        assert np.max(np.abs(h - want)) < 1e-11


class TestALongListCostsItsArrivalsNotItsSpan:
    """Each arrival's impulse is built on a record of fixed length
    (2 × 4096 + 1 samples), so a list costs its arrivals times that, not
    its arrivals times its delay span: 1000 arrivals over 4 s at 48 kHz
    take about 1 s through a reception and 1.7 s into an impulse response,
    where an impulse per arrival over the whole record took 12.5 s and
    10.6 s. The same 300 arrivals spread over 4 s cost about what they do
    over 0.25 s."""

    FS = 48000.0

    @staticmethod
    def _seconds(call):
        import time
        best = np.inf
        for _ in range(2):
            begin = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - begin)
        return best

    @pytest.mark.parametrize('route', ['reception', 'impulse_response'])
    def test_spreading_the_arrivals_does_not_multiply_the_cost(self, route):
        from uacpy.acoustic_signal import (arrival_impulse_response,
                                           simulate_arrival_reception)
        rng = np.random.default_rng(0)
        unit = np.sort(rng.uniform(0.0, 1.0, 300))
        amps, phases = rng.uniform(0.1, 1.0, 300), rng.uniform(-3, 3, 300)
        pulse = (np.hanning(480)
                 * np.sin(2 * np.pi * 3000.0 * np.arange(480) / self.FS))

        def run(span):
            delays = 2.0 + span * unit
            if route == 'reception':
                return lambda: simulate_arrival_reception(
                    amps, delays, source_waveform=pulse, sample_rate=self.FS,
                    trace_frequency=3000.0, phases_rad=phases)
            return lambda: arrival_impulse_response(
                amps, delays, sample_rate=self.FS, phases_rad=phases)
        assert self._seconds(run(4.0)) < 3.0 * self._seconds(run(0.25))


class TestAStartAfterTheArrivalsIsRefused:
    """A stated start after every arrival, with the record left to the
    arrivals, leaves nothing to size it from: the call is refused by name,
    saying where the arrivals end, not left to a negative array size."""

    WF = np.hanning(64)
    CELL = {'amplitudes': [1.0], 'delays': [0.1]}

    def _call(self, name, t_start):
        from uacpy.acoustic_signal import (arrival_impulse_response,
                                           delayandsum, simulate_arrival_grid,
                                           simulate_arrival_reception)
        kw = dict(source_waveform=self.WF, sample_rate=8000.0,
                  trace_frequency=1000.0, t_start=t_start)
        return {
            'arrival_impulse_response': lambda: arrival_impulse_response(
                [1.0], [0.1], sample_rate=8000.0, t_start=t_start),
            'simulate_arrival_reception': lambda: simulate_arrival_reception(
                [1.0], [0.1], **kw),
            'simulate_arrival_grid': lambda: simulate_arrival_grid(
                [[self.CELL]], **kw),
            'delayandsum': lambda: delayandsum(self.CELL, **kw),
        }[name]

    @pytest.mark.parametrize('name, past', [
        ('arrival_impulse_response', 0.14),
        ('simulate_arrival_reception', 0.11),
        ('simulate_arrival_grid', 0.11),
        ('delayandsum', 0.11)])
    def test_a_start_past_them_is_refused_by_name(self, name, past):
        with pytest.raises(ConfigurationError,
                           match=f"^{name}: t_start.* is after the last"):
            self._call(name, past)()

    @pytest.mark.parametrize('name, inside', [
        ('arrival_impulse_response', 0.12),
        ('simulate_arrival_reception', 0.105),
        ('simulate_arrival_grid', 0.105),
        ('delayandsum', 0.105)])
    def test_a_start_inside_them_is_taken(self, name, inside):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = self._call(name, inside)()
        assert np.size(out[1]) > 0

    def test_a_start_per_cell_names_the_cells_and_a_remedy(self):
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^impulse_response_from_transfer_function"
                                 r": t_start must be finite.*one per cell of"
                                 r" H"):
            impulse_response_from_transfer_function(
                np.ones(5), frequencies=np.arange(5.0) * 100.0,
                sample_rate=1000.0, t_start=None)

    def test_one_list_takes_one_start(self):
        from uacpy.acoustic_signal import arrival_impulse_response
        with pytest.raises(ConfigurationError,
                           match="^arrival_impulse_response: t_start must "
                                 "be one finite time"):
            arrival_impulse_response([1.0], [0.1], sample_rate=8000.0,
                                     t_start='reduced')


class TestAReceptionWindowHasALength:
    """A reception's stated ``output_duration`` is a positive, finite number
    of seconds; anything else is refused by the call's name, not left to a
    negative array size or an empty record."""

    WF = np.hanning(64)
    CELL = {'amplitudes': [1.0], 'delays': [0.1]}

    def _call(self, name, duration):
        from uacpy.acoustic_signal import (simulate_arrival_grid,
                                           simulate_arrival_reception)
        kw = dict(source_waveform=self.WF, sample_rate=8000.0,
                  trace_frequency=1000.0, output_duration=duration)
        if name == 'simulate_arrival_reception':
            return simulate_arrival_reception([1.0], [0.1], **kw)
        return simulate_arrival_grid([[self.CELL]], **kw)

    @pytest.mark.parametrize('name', ['simulate_arrival_reception',
                                      'simulate_arrival_grid'])
    @pytest.mark.parametrize('duration', [0.0, -1.0, np.nan, 'x', [0.1],
                                          True])
    def test_no_length_is_refused_saying_what_to_pass(self, name, duration):
        with pytest.raises(ConfigurationError,
                           match=f"^{name}: output_duration must be.*Pass "
                                 f"the window's length"):
            self._call(name, duration)

    @pytest.mark.parametrize('name', ['simulate_arrival_reception',
                                      'simulate_arrival_grid'])
    def test_the_shortest_length_is_one_sample(self, name):
        t, y = self._call(name, 1e-5)
        assert np.shape(y)[-1] == 1


class TestAnArrivalAtARecordsEndKeepsItsWeight:
    """A stated record's taper runs from the samples bracketing its first
    and last arrival: an arrival a hair inside the first or the last sample
    — a start one ulp early, a delay read back through float32 — keeps its
    tap and its energy, as one on the sample does. The taper itself is two
    half windows written into ones, a little over twice the window's own
    bytes at its peak."""

    FS = 8000.0

    @pytest.mark.parametrize('offset', [0.0, 1e-12, 1e-6])
    @pytest.mark.parametrize('end', ['first', 'last'])
    def test_an_arrival_just_inside_an_end_keeps_its_tap(self, end, offset):
        from uacpy.acoustic_signal import arrival_impulse_response
        position = offset if end == 'first' else 99.0 - offset
        _, h = arrival_impulse_response([1.0], [0.1], sample_rate=self.FS,
                                        t_start=0.1 - position / self.FS,
                                        n_samples=100)
        tap = h[0] if end == 'first' else h[-1]
        assert tap > 0.999 and np.sum(h ** 2) > 0.999

    @pytest.mark.parametrize('end', ['first', 'last'])
    def test_an_arrival_inside_an_end_sample_keeps_its_neighbouring_tap(
            self, end):
        """0.7 samples from either end of the record, the arrival's
        band-limited impulse puts sinc(0.7) = 0.37 of it on the end
        sample."""
        from uacpy.acoustic_signal import arrival_impulse_response
        position = 0.7 if end == 'first' else 99.0 - 0.7
        _, h = arrival_impulse_response([1.0], [0.1], sample_rate=self.FS,
                                        t_start=0.1 - position / self.FS,
                                        n_samples=100)
        tap = h[0] if end == 'first' else h[-1]
        assert tap == pytest.approx(np.sinc(0.7), abs=2e-3)

    def test_the_taper_is_one_on_the_samples_bracketing_the_arrivals(
            self, monkeypatch):
        """The taper runs from floor(first) and ceil(last), the samples
        either side of the first and last arrival — the nearest samples
        would leave the neighbour of an arrival past half a sample on the
        ramp — and a position the rounding of the times moves off a sample
        is on it: at 1000 s and 48 kHz the record sized from the arrivals
        puts its first arrival 255.9999999976 samples in, its 256-sample
        margin."""
        from uacpy.acoustic_signal import channel
        spans = []
        taper = channel._arrival_record_taper

        def spy(first, last, n_samples):
            if np.ndim(first) == 0:
                spans.append((float(first), float(last)))
            return taper(first, last, n_samples)

        monkeypatch.setattr(channel, '_arrival_record_taper', spy)
        channel.arrival_impulse_response(
            [1.0, 0.5], 0.1 + np.array([0.0, 200.6]) / self.FS,
            sample_rate=self.FS, t_start=0.1 - 300.7 / self.FS,
            n_samples=900)
        channel.arrival_impulse_response([1.0, 0.5], [1000.0, 1000.00005],
                                         sample_rate=48000.0)
        assert spans == [(300.0, 502.0), (256.0, 259.0)]

    def test_a_start_one_ulp_early_keeps_the_arrival(self):
        from uacpy.acoustic_signal import arrival_impulse_response
        _, h = arrival_impulse_response([1.0], [0.1], sample_rate=self.FS,
                                        t_start=np.nextafter(0.1, 0.0),
                                        n_samples=2)
        np.testing.assert_allclose(h, [1.0, 0.0], atol=1e-9)

    def test_a_one_sample_record_holds_its_arrival(self):
        from uacpy.acoustic_signal import (arrival_grid_impulse_response,
                                           arrival_impulse_response)
        _, h = arrival_impulse_response([1.0], [0.1], sample_rate=self.FS,
                                        t_start=0.1, n_samples=1)
        _, grid = arrival_grid_impulse_response(
            [[{'amplitudes': [1.0], 'delays': [0.1]}]], sample_rate=self.FS,
            t_start=0.1, n_samples=1)
        np.testing.assert_allclose(h, [1.0])
        np.testing.assert_allclose(grid, [[[1.0]]])

    def test_the_record_taper_holds_little_beside_its_window(self):
        import tracemalloc
        from uacpy.acoustic_signal.channel import _arrival_record_taper
        n = 2_880_000
        tracemalloc.start()
        try:
            window = _arrival_record_taper(960_000.0, 1_920_000.0, n)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert window[0] == 0.0 and window[1_500_000] == 1.0
        assert peak < 2.5 * window.nbytes


class TestATwoSampleRecordHoldsItsArrival:
    """An arrival on a record of two samples is the tap on its sample, and
    one between them its band-limited impulse on both: the record is built
    padded to the margin, so the arrival's kernel is not cut to the
    record."""

    def test_the_arrival_is_its_tap(self):
        # The start's own rounding puts the arrival 1.4e-14 samples off
        # sample 1, and sinc puts that much on sample 0.
        from uacpy.acoustic_signal import arrival_impulse_response
        t, h = arrival_impulse_response([1.0], [0.1], sample_rate=8000.0,
                                        t_start=0.1 - 1.0 / 8000.0,
                                        n_samples=2)
        np.testing.assert_allclose(h, [0.0, 1.0], atol=1e-13)

    @pytest.mark.parametrize('n_samples', [2, 3, 8])
    def test_an_arrival_between_samples_is_its_impulse(self, n_samples):
        from uacpy.acoustic_signal import arrival_impulse_response
        position = (n_samples - 1) / 2.0 + 0.25
        t, h = arrival_impulse_response([1.0], [0.1], sample_rate=8000.0,
                                        t_start=0.1 - position / 8000.0,
                                        n_samples=n_samples)
        np.testing.assert_allclose(
            h, np.sinc(np.arange(n_samples) - position), atol=5e-3)


class TestAStatedRecordHoldsTheResponsesOwnSamples:
    """A record whose start or end the caller states closer to the arrivals
    than the margin is built padded to the margin and cut back: its taps
    are the band-limited response's own samples — the precursor of a path
    between samples, the quadrature tail of a phase-rotated one — as on the
    record sized from the arrivals (built on the stated record alone, a
    start 1.3 samples before a path at pi/2 takes 0.54 of the peak from its
    taps)."""

    FS = 8000.0
    DELAYS = 0.1 + np.array([0.0, 5.0, 9.7]) / 8000.0
    AMPS = np.array([1.0, 0.6, 0.4])

    @staticmethod
    def _true(amps, positions, phases, n_samples):
        """The lossless band-limited response on the record's samples:
        ``cos φ·sinc(x) − sin φ·(1 − cos πx)/(πx)`` per path."""
        x = np.arange(n_samples)[:, None] - positions[None, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            hilbert = np.where(np.abs(x) < 1e-12, 0.0,
                               (1.0 - np.cos(np.pi * x)) / (np.pi * x))
        return (np.sinc(x) * amps * np.cos(phases)
                - hilbert * amps * np.sin(phases)).sum(axis=1)

    def _error(self, phases, t_start, n_samples):
        from uacpy.acoustic_signal import arrival_impulse_response
        t, h = arrival_impulse_response(
            self.AMPS, self.DELAYS, phases_rad=phases, sample_rate=self.FS,
            t_start=t_start, n_samples=n_samples)
        positions = (self.DELAYS - t[0]) * self.FS
        truth = self._true(self.AMPS, positions, phases, n_samples)
        near = ((np.arange(n_samples) >= positions[0] - 3)
                & (np.arange(n_samples) <= positions[-1] + 3))
        return np.max(np.abs(h - truth)[near]) / np.max(np.abs(truth))

    @pytest.mark.parametrize('phase', [0.0, np.pi / 2])
    @pytest.mark.parametrize('early', [0.3, 1.3, 4.3])
    def test_a_start_close_before_the_first_arrival(self, phase, early):
        phases = np.array([phase, np.pi, 1.2])
        assert self._error(phases, self.DELAYS[0] - early / self.FS,
                           64) < 1e-2

    @pytest.mark.parametrize('phase', [0.0, np.pi / 2])
    @pytest.mark.parametrize('late', [0.3, 1.3, 4.3])
    def test_an_end_close_after_the_last_arrival(self, phase, late):
        phases = np.array([1.2, np.pi, phase])
        t_start = self.DELAYS[0] - 40.0 / self.FS
        n_samples = int(round(40.0 + 9.7 + late)) + 1
        assert self._error(phases, t_start, n_samples) < 1e-2


class TestTheArrivalFunctionsRefuseWhatTheyCannotRead:
    """A record length that is not a whole number of samples, a band that is
    not two numbers inside [0, sample_rate/2] and a phase offset that is not
    one finite angle are refused by the call's name with what to pass —
    a bool is not a count or an angle, though it would read as 1 or 0."""

    CELL = {'amplitudes': [1.0, 0.5], 'delays': [0.1, 0.1031]}

    def _plain(self, **kw):
        from uacpy.acoustic_signal import arrival_impulse_response
        return arrival_impulse_response([1.0, 0.5], [0.1, 0.1031],
                                        sample_rate=8000.0, **kw)

    def _grid(self, **kw):
        from uacpy.acoustic_signal import arrival_grid_impulse_response
        return arrival_grid_impulse_response([[self.CELL]],
                                             sample_rate=8000.0, **kw)

    @pytest.mark.parametrize('n_samples', ['x', [5], np.nan, True, 2.5, 0])
    @pytest.mark.parametrize('call', ['_plain', '_grid'])
    def test_a_record_length_that_is_not_a_count_is_refused(self, call,
                                                            n_samples):
        with pytest.raises(ConfigurationError,
                           match=r"(?s)_impulse_response: n_samples must be "
                                 r"a whole number.*How to fix.*samples"):
            getattr(self, call)(n_samples=n_samples)

    def test_a_whole_float_is_a_record_length(self):
        t, h = self._plain(n_samples=600.0)
        assert h.size == 600

    @pytest.mark.parametrize('band', ['ab', (1.0, 2.0, 3.0), 5.0,
                                      (np.nan, 100.0), np.array([False, True]),
                                      (True, 1000.0), (0.0, np.True_),
                                      (0.0, 100.0 + 1j), (0.0, 5000.0),
                                      (300.0, 300.0)])
    @pytest.mark.parametrize('call', ['_plain', '_grid'])
    def test_a_band_that_is_not_two_numbers_in_range_is_refused(self, call,
                                                                band):
        with pytest.raises(ConfigurationError,
                           match=r"(?s)_impulse_response: band=.* must be two "
                                 r"numbers.*How to fix.*None"):
            getattr(self, call)(band=band)

    @pytest.mark.parametrize('offset', ['x', np.nan, [0.1], True, 1j,
                                        np.array(True)])
    @pytest.mark.parametrize('name', [
        'arrival_transfer_function', 'arrival_grid_transfer_function',
        'arrival_impulse_response', 'arrival_grid_impulse_response',
        'simulate_arrival_reception', 'delayandsum'])
    def test_a_phase_offset_that_is_not_one_angle_is_refused(self, name,
                                                             offset):
        from uacpy import acoustic_signal as sig
        wf = np.hanning(64) * np.sin(2 * np.pi * 1000.0 * np.arange(64)
                                     / 8000.0)
        amps, delays = [1.0, 0.5], [0.1, 0.1031]
        calls = {
            'arrival_transfer_function': lambda: sig.arrival_transfer_function(
                [100.0], amps, delays, phase_offset=offset),
            'arrival_grid_transfer_function':
                lambda: sig.arrival_grid_transfer_function(
                    [100.0], [[self.CELL]], phase_offset=offset),
            'arrival_impulse_response': lambda: self._plain(
                phase_offset=offset),
            'arrival_grid_impulse_response': lambda: self._grid(
                phase_offset=offset),
            'simulate_arrival_reception':
                lambda: sig.simulate_arrival_reception(
                    amps, delays, source_waveform=wf, sample_rate=8000.0,
                    trace_frequency=1000.0, phase_offset=offset),
            'delayandsum': lambda: sig.delayandsum(
                self.CELL, source_waveform=wf, sample_rate=8000.0,
                trace_frequency=1000.0, phase_offset=offset)}
        with pytest.raises(ConfigurationError,
                           match=rf"(?s)^{name}: phase_offset must be one "
                                 rf"finite angle.*How to fix"):
            calls[name]()


class TestAnArrivalColumnHoldsRealNumbers:
    """Each column of an arrival list — amplitudes, delays, their imaginary
    parts, phases — holds one real number per arrival: a complex phase, a
    bool or text among them is refused by the call's name, where a cast
    would raise a raw TypeError or read 1, its real part, or what the text
    spells."""

    WF = np.hanning(64) * np.sin(2 * np.pi * 1000.0 * np.arange(64) / 8000.0)

    @pytest.mark.parametrize('column, values', [
        ('phases_rad', [0.5 + 0.1j, 0.0]), ('phases_rad', [True, 0.0]),
        ('delays_imag_s', [0.0, '1e-6']), ('delays_s', [0.1, True]),
        ('amplitudes', ['1.0', 0.5])])
    @pytest.mark.parametrize('name', ['arrival_transfer_function',
                                      'arrival_impulse_response',
                                      'simulate_arrival_reception'])
    def test_a_column_that_is_not_real_numbers_is_refused(self, name, column,
                                                          values):
        from uacpy import acoustic_signal as sig
        kw = dict(amplitudes=[1.0, 0.5], delays_s=[0.1, 0.1031])
        kw[column] = values
        amps, delays = kw.pop('amplitudes'), kw.pop('delays_s')
        calls = {
            'arrival_transfer_function': lambda: sig.arrival_transfer_function(
                [100.0], amps, delays, **kw),
            'arrival_impulse_response': lambda: sig.arrival_impulse_response(
                amps, delays, sample_rate=8000.0, **kw),
            'simulate_arrival_reception':
                lambda: sig.simulate_arrival_reception(
                    amps, delays, source_waveform=self.WF, sample_rate=8000.0,
                    trace_frequency=1000.0, **kw)}
        with pytest.raises(ConfigurationError,
                           match=rf"(?s)^{name}: {column} must be real "
                                 rf"numbers, one per arrival.*How to fix"):
            calls[name]()


class TestAPerCellStartReadsEachCell:
    """A start per cell is read by the package's one rule cell by cell: a
    bool among real starts (NumPy reads ``[True, 0.05]`` as numbers) is
    refused, a list of real starts taken."""

    CELL = {'amplitudes': [1.0], 'delays': [0.1]}

    def test_a_bool_among_the_starts_is_refused(self):
        from uacpy.acoustic_signal import arrival_grid_impulse_response
        with pytest.raises(ConfigurationError,
                           match="^arrival_grid_impulse_response: t_start "
                                 "must be finite times"):
            arrival_grid_impulse_response([[self.CELL, self.CELL]],
                                          sample_rate=8000.0,
                                          t_start=[[True, 0.05]])

    def test_real_starts_are_each_cells_start(self):
        from uacpy.acoustic_signal import arrival_grid_impulse_response
        t, _ = arrival_grid_impulse_response([[self.CELL, self.CELL]],
                                             sample_rate=8000.0,
                                             t_start=[[0.05, 0.06]])
        np.testing.assert_allclose(t[0, :, 0], [0.05, 0.06])


class TestABandIsReadOnTheRecordTheResponseIsBuiltOn:
    """A stated record closer to its arrivals than the margin is built on
    the padded record, so a band is counted on that record's grid: a
    64-sample record at 48 kHz (a 750 Hz grid) takes band=(100, 500) Hz,
    which the 513-sample record it is built on (a 93.6 Hz grid) resolves
    in four bins."""

    def test_a_band_finer_than_the_stated_grid_is_built(self):
        from uacpy.acoustic_signal import arrival_impulse_response
        fs = 48000.0
        t, h = arrival_impulse_response([1.0], [0.01], sample_rate=fs,
                                        t_start=0.01 - 10.0 / fs,
                                        n_samples=64, band=(100.0, 500.0))
        assert h.size == 64 and np.all(np.isfinite(h))
        assert np.max(np.abs(h)) > 0.0

    def test_a_band_the_built_record_cannot_hold_is_refused(self):
        from uacpy.acoustic_signal import arrival_impulse_response
        fs = 48000.0
        with pytest.raises(ConfigurationError,
                           match=r"band=\(100, 101\) Hz keeps 0 bin\(s\) of "
                                 r"the 93\.5673 Hz grid of the 513-sample "
                                 r"record the response is built on"):
            arrival_impulse_response([1.0], [0.01], sample_rate=fs,
                                     t_start=0.01 - 10.0 / fs, n_samples=64,
                                     band=(100.0, 101.0))


class TestEveryArrivalCellFunctionTakesOneCell:
    """A cell holding only ``amplitudes`` and ``delays`` — no
    ``delays_imag``, ``phases`` or ``n_arrivals``, which default to zeros
    and the list's length — is taken by every function that takes arrival
    cells, and gives what the plain functions give; one holding one column
    and not the other, columns of different lengths, or a count its
    columns contradict is refused, not read as a receiver nothing reaches
    or a list missing arrivals. A grid's refusal names the cell."""

    WF = np.hanning(64) * np.sin(2 * np.pi * 1000.0 * np.arange(64) / 8000.0)
    CELL = {'amplitudes': [1.0, 0.5], 'delays': [0.1, 0.1031]}

    def _plain(self):
        from uacpy.acoustic_signal import simulate_arrival_reception
        return simulate_arrival_reception(
            [1.0, 0.5], [0.1, 0.1031], source_waveform=self.WF,
            sample_rate=8000.0, trace_frequency=1000.0)

    def test_the_grid_takes_it(self):
        from uacpy.acoustic_signal import simulate_arrival_grid
        t, traces = simulate_arrival_grid(
            [[self.CELL]], source_waveform=self.WF, sample_rate=8000.0,
            trace_frequency=1000.0)
        t0, y0 = self._plain()
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(traces[0, 0], y0)

    def test_the_transfer_function_grid_takes_it(self):
        from uacpy.acoustic_signal import (arrival_grid_transfer_function,
                                           arrival_transfer_function)
        f = np.array([100.0, 250.0, 400.0])
        H = arrival_grid_transfer_function(f, [[self.CELL]])
        np.testing.assert_array_equal(
            H[0, 0], arrival_transfer_function(f, [1.0, 0.5], [0.1, 0.1031]))

    @pytest.mark.parametrize('cell, said', [
        ({'amplitudes': [1.0]}, "holds arrivals but no 'delays'"),
        ({'delays': [0.1]}, "holds arrivals but no 'amplitudes'"),
        ({'amplitudes': [1.0], 'delays': []},
         "holds 1 amplitude\\(s\\) and 0 delay\\(s\\)"),
        ({'n_arrivals': 1, 'amplitudes': [1.0, 0.5], 'delays': [0.1]},
         "holds 2 amplitude\\(s\\) and 1 delay\\(s\\)"),
        ({'n_arrivals': 0, 'amplitudes': [1.0], 'delays': [0.1]},
         "states n_arrivals=0 but holds 1 arrival\\(s\\)"),
        ({'n_arrivals': 3, 'amplitudes': [1.0], 'delays': [0.1]},
         "states n_arrivals=3 but holds 1 arrival\\(s\\)"),
        ({'n_arrivals': 'x', 'amplitudes': [1.0], 'delays': [0.1]},
         "states n_arrivals='x', which is not a count"),
        ({'n_arrivals': True, 'amplitudes': [1.0], 'delays': [0.1]},
         "states n_arrivals=True, which is not a count"),
        ({'n_arrivals': 1.5, 'amplitudes': [1.0], 'delays': [0.1]},
         "states n_arrivals=1.5, which is not a count"),
        ([1.0, 0.1], "is a list, not a mapping of columns")])
    @pytest.mark.parametrize('name', ['simulate_arrival_grid', 'delayandsum',
                                      'arrival_grid_transfer_function',
                                      'arrival_grid_impulse_response'])
    def test_a_cell_that_is_not_one_arrival_list_is_refused(self, name, cell,
                                                             said):
        from uacpy import acoustic_signal as sig
        at = '' if name == 'delayandsum' else r' at cells\[0\]\[0\]'
        calls = {
            'simulate_arrival_grid': lambda: sig.simulate_arrival_grid(
                [[cell]], source_waveform=self.WF, sample_rate=8000.0,
                trace_frequency=1000.0),
            'delayandsum': lambda: sig.delayandsum(
                cell, source_waveform=self.WF, sample_rate=8000.0,
                trace_frequency=1000.0),
            'arrival_grid_transfer_function':
                lambda: sig.arrival_grid_transfer_function([100.0], [[cell]]),
            'arrival_grid_impulse_response':
                lambda: sig.arrival_grid_impulse_response(
                    [[cell]], sample_rate=8000.0)}
        with pytest.raises(ConfigurationError,
                           match=f"^{name}: an arrival cell{at} {said}"):
            calls[name]()

    def test_a_grid_refusal_names_the_cell(self):
        from uacpy.acoustic_signal import arrival_grid_impulse_response
        cells = [[self.CELL, self.CELL], [self.CELL, {'delays': [0.1]}]]
        with pytest.raises(ConfigurationError,
                           match=r"^arrival_grid_impulse_response: an "
                                 r"arrival cell at cells\[1\]\[1\] holds"):
            arrival_grid_impulse_response(cells, sample_rate=8000.0)

    def test_an_empty_cell_is_a_receiver_nothing_reaches(self):
        from uacpy.acoustic_signal import arrival_grid_transfer_function
        H = arrival_grid_transfer_function([100.0], [[{}]])
        assert np.all(H == 0)

    def test_delayandsum_takes_it(self):
        from uacpy.acoustic_signal import delayandsum
        y, t = delayandsum(self.CELL, source_waveform=self.WF,
                           sample_rate=8000.0, trace_frequency=1000.0)
        t0, y0 = self._plain()
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)


class TestAnArrivalListIsSummedInBoundedMemory:
    """An arrival list is summed in blocks, so its scratch does not grow
    with it: the transfer function over blocks of arrivals (200 lossy
    arrivals over 36 001 lines took 290 MB as one outer product), the
    kernels of a reception or an impulse response a block of kernel samples
    at a time — 400 arrivals over 2 s at 24 kHz hold about 40 MB, as 2000
    do, where whole-list blocks of kernels held about 120 MB."""

    FS = 24000.0

    def _list(self, n, span):
        rng = np.random.default_rng(0)
        delays = 30.0 + np.sort(rng.uniform(0.0, span, n))
        return dict(amplitudes=rng.uniform(0.1, 1.0, n), delays_s=delays,
                    phases_rad=rng.uniform(-np.pi, np.pi, n),
                    delays_imag_s=-rng.uniform(0.0, 1e-5, n))

    @pytest.mark.parametrize('route, bound', [
        ('reception', 60e6), ('impulse_response', 60e6),
        ('transfer_function', 100e6)])
    def test_a_long_list_over_a_long_span_stays_bounded(self, route, bound):
        import tracemalloc
        from uacpy.acoustic_signal import (arrival_impulse_response,
                                           arrival_transfer_function,
                                           simulate_arrival_reception)
        arrivals = (self._list(200, 1.0) if route == 'transfer_function'
                    else self._list(400, 2.0))
        amps, delays = arrivals.pop('amplitudes'), arrivals.pop('delays_s')
        pulse = (np.hanning(480)
                 * np.sin(2 * np.pi * 3000.0 * np.arange(480) / self.FS))
        tracemalloc.start()
        try:
            if route == 'reception':
                simulate_arrival_reception(
                    amps, delays, source_waveform=pulse, sample_rate=self.FS,
                    trace_frequency=3000.0, **arrivals)
            elif route == 'impulse_response':
                arrival_impulse_response(amps, delays, sample_rate=self.FS,
                                         trace_frequency=3000.0, **arrivals)
            else:
                arrival_transfer_function(
                    np.linspace(0.0, 12000.0, 36001), amps, delays,
                    trace_frequency=3000.0, **arrivals)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < bound

    def test_a_list_longer_than_one_block_sums_every_arrival(self):
        # 200 arrivals over 20 001 lines is two blocks of the sum: every
        # arrival is in it, once, as the arrival-by-arrival closed form.
        from uacpy.acoustic_signal import arrival_transfer_function
        from uacpy.acoustic_signal.channel import _ARRIVAL_SUM_BLOCK_ELEMS
        f = np.linspace(0.0, 10000.0, 20001)
        arrivals = self._list(200, 1.0)
        assert 200 * f.size > _ARRIVAL_SUM_BLOCK_ELEMS
        H = arrival_transfer_function(f, arrivals['amplitudes'],
                                      arrivals['delays_s'],
                                      delays_imag_s=arrivals['delays_imag_s'],
                                      phases_rad=arrivals['phases_rad'])
        omega = 2 * np.pi * f
        closed = sum(
            a * np.exp(1j * ph) * np.exp(omega * ti) * np.exp(-1j * omega * tr)
            for a, tr, ti, ph in zip(arrivals['amplitudes'],
                                     arrivals['delays_s'],
                                     arrivals['delays_imag_s'],
                                     arrivals['phases_rad']))
        assert np.max(np.abs(H - closed)) < 1e-12 * np.max(np.abs(closed))


class TestAnArrivalRecordKeepsItsSpanEitherSide:
    """An arrival list's own record keeps a margin either side of its
    arrivals: their delay span, and at least 256 samples (room for the tail
    of a phase-rotated path to fall away)."""

    FS = 1000.0

    @pytest.mark.parametrize('span_samples, margin_samples', [
        (255, 256), (256, 256), (257, 257), (400, 400)])
    def test_the_margin_is_the_span_and_at_least_256_samples(
            self, span_samples, margin_samples):
        from uacpy.acoustic_signal import arrival_impulse_response
        first = 2.0
        last = first + span_samples / self.FS
        t, h = arrival_impulse_response([1.0, 0.5], [first, last],
                                        sample_rate=self.FS)
        assert t[0] == pytest.approx(first - margin_samples / self.FS,
                                     abs=1e-9)
        assert t[-1] == pytest.approx(last + margin_samples / self.FS,
                                      abs=1.0 / self.FS)


class TestARecordThatMissesAnArrivalLeavesItOut:
    """An arrival impulse response on a stated record holds only the
    arrivals on it: an arrival outside it would leave on it only a cut
    fragment of its impulse, its phase rotation wrapped round the record, so
    it is left out and a ``NumericsWarning`` names the energy left out. An
    arrival on the record's first or last sample is on it."""

    FS = 1000.0

    def _response(self, delays, **kw):
        from uacpy.acoustic_signal import arrival_impulse_response
        amps = [1.0, 0.5][:len(delays)]
        phases = [0.0, 1.0][:len(delays)]
        return arrival_impulse_response(amps, delays, sample_rate=self.FS,
                                        phases_rad=phases, t_start=0.0,
                                        n_samples=100, **kw)

    @pytest.mark.parametrize('delay', [0.0, 0.099])
    def test_an_arrival_on_the_records_end_samples_is_held(self, delay):
        with recorded_warnings() as record:
            _, h = self._response([0.05, delay])
        assert not record, [str(w.message) for w in record]
        assert np.max(np.abs(h)) > 0.0

    def test_the_share_left_out_is_weighed_with_its_absorption(self):
        # Equal magnitudes; the one left out is absorbed by exp(2 pi 1000 Hz
        # * -2e-4 s) at the trace frequency, so it carries 0.081 of the
        # other's power: -11 dB of the received energy, where the magnitudes
        # alone would say -3 dB.
        from uacpy.acoustic_signal import arrival_impulse_response
        with recorded_warnings() as record:
            arrival_impulse_response(
                [1.0, 1.0], [0.05, 0.15], sample_rate=self.FS,
                delays_imag_s=[0.0, -2e-4], trace_frequency=1000.0,
                t_start=0.0, n_samples=100)
        messages = [str(w.message) for w in record
                    if issubclass(w.category, NumericsWarning)]
        assert len(messages) == 1, messages
        assert '-11 dB of the received energy' in messages[0]

    @pytest.mark.parametrize('delay', [-0.0005, 0.0995, 0.15])
    def test_an_arrival_off_the_record_is_left_out_and_said(self, delay):
        with recorded_warnings() as record:
            _, h = self._response([0.05, delay])
        _, alone = self._response([0.05])
        messages = [str(w.message) for w in record
                    if issubclass(w.category, NumericsWarning)]
        assert len(messages) == 1, messages
        assert messages[0].startswith('arrival_impulse_response: the record')
        # 0.5**2 of 1.25: the omitted share, in dB.
        assert '-7 dB of the received energy' in messages[0]
        assert 'n_samples=' in messages[0]
        np.testing.assert_allclose(h, alone, atol=1e-15)


class TestEchoesLandAtTheDelayTheyWereGiven:
    """COA eq. 8.30 places each echo at ``S[t - tau]`` — a CONTINUOUS shift.

    Rounding tau to the nearest sample is a numerical convenience the
    formulation does not license: it moves every echo by up to half a
    sample, which is a phase error of tens of degrees at any carrier a
    modem would use, and ``delayandsum`` sums the echoes coherently, so
    that error lands in the interference pattern.
    """

    FS = 200000.0
    FC = 20000.0

    def _cell(self, delay):
        return dict(n_arrivals=1,
                    amplitudes=np.array([1.0]),
                    phases=np.array([0.0]),
                    delays=np.array([float(delay)]),
                    delays_imag=np.array([0.0]),
                    n_top_bounces=np.array([0]),
                    n_bot_bounces=np.array([0]),
                    source_angles=np.array([0.0]),
                    receiver_angles=np.array([0.0]))

    @staticmethod
    def _waveform(fs, fc, n=256):
        k = np.arange(n)
        return np.hanning(n) * np.sin(2 * np.pi * fc * k / fs)

    def _trace(self, delay, **kw):
        from uacpy.acoustic_signal import delayandsum
        rts, _ = delayandsum(
            rcv_arrivals=self._cell(delay),
            source_waveform=self._waveform(self.FS, self.FC),
            sample_rate=self.FS, trace_frequency=self.FC,
            t_start=0.0, output_duration=0.02, **kw)
        return rts

    def test_a_half_sample_echo_is_not_snapped_onto_a_sample(self):
        """An echo half a sample late must land between the two samples, not
        on one of them. Rounding makes the half-sample trace bit-identical
        to a whole-sample one, which is how the error hides."""
        dt = 1.0 / self.FS
        base = 1000 * dt
        half = self._trace(base + 0.5 * dt)
        assert not np.allclose(half, self._trace(base), atol=1e-12), (
            "the half-sample echo was snapped back onto sample 1000")
        assert not np.allclose(half, self._trace(base + dt), atol=1e-12), (
            "the half-sample echo was snapped forward onto sample 1001")

    def test_the_echo_arrives_where_it_was_told_to(self):
        """Measured as the energy centroid of the trace, which reads the
        group delay whatever the waveform. Rounding quantises it to the
        sample grid; the requested delay does not sit on that grid."""
        dt = 1.0 / self.FS
        for frac in (0.25, 0.5, 0.75):
            want = (1000 + frac) * dt
            rts = self._trace(want)
            power = np.abs(rts) ** 2
            centroid = float((power * np.arange(power.size)).sum()
                             / power.sum()) * dt
            reference = self._trace(1000 * dt)
            p0 = np.abs(reference) ** 2
            c0 = float((p0 * np.arange(p0.size)).sum() / p0.sum()) * dt
            shift_samples = (centroid - c0) / dt
            assert shift_samples == pytest.approx(frac, abs=0.02), (
                f"asked for +{frac} samples, the trace moved "
                f"{shift_samples:.3f}")

    def test_nearest_sample_placement_stays_available(self):
        """The old behaviour is a keyword, not a removal: a caller who wants
        every echo on a sample boundary can still have it. 0.6 rather than
        0.5 of a sample, because numpy rounds a half to the nearest EVEN
        sample, so 1000.5 snaps down to 1000 and would not show the move."""
        dt = 1.0 / self.FS
        snapped = self._trace(1000.6 * dt, fractional=False)
        assert np.allclose(snapped, self._trace(1001 * dt, fractional=False),
                           atol=1e-12)
        assert not np.allclose(snapped, self._trace(1000 * dt,
                                                   fractional=False),
                               atol=1e-12)


class TestVolumeAttenuationFromImaginaryDelay:
    """``Im(tau)`` is the only carrier of Bellhop's volume attenuation.

    ``ArrMod.f90:118-125`` writes ``REAL(delay)`` and ``AIMAG(delay)`` as
    separate fields and the amplitude column does **not** include the loss;
    ``delayandsum.m:134`` applies ``exp(omega*Im(tau))`` as its own factor.
    Dropping it returns a silently lossless field — -23 dB at 20 kHz.
    """

    TAU_I = -1.83e-5          # s; -20.0 dB at 20 kHz
    FC = 20000.0

    def _cell(self):
        return dict(n_arrivals=1,
                    amplitudes=np.array([1.0]),
                    phases=np.array([0.0]),
                    delays=np.array([1.0]),
                    delays_imag=np.array([self.TAU_I]),
                    n_top_bounces=np.array([0]),
                    n_bot_bounces=np.array([0]),
                    source_angles=np.array([0.0]),
                    receiver_angles=np.array([0.0]))

    def test_transfer_function_applies_it(self):
        H = _cell_tf(self._cell(), np.array([self.FC]))
        expected = np.exp(2 * np.pi * self.FC * self.TAU_I)
        assert np.abs(np.asarray(H).ravel()[0]) == pytest.approx(expected,
                                                                 rel=1e-9)
        assert 20 * np.log10(expected) < -19.0

    def test_delay_and_sum_applies_it_at_every_frequency(self):
        """Each frequency of the pulse is absorbed by its own
        ``exp(2 pi f Im tau)`` (linear in f with no law given), not by the
        one factor at fc: the 16 and 24 kHz components of a 20 kHz pulse
        lose 16 and 24 dB, as the transfer function says."""
        from uacpy.acoustic_signal import delayandsum
        fs = 200000.0
        n = 2000
        t = np.arange(n) / fs
        # A 10-30 kHz chirp, so every checked frequency carries energy.
        wf = np.hanning(n) * np.sin(
            2 * np.pi * (10000.0 * t + 0.5 * (20000.0 / t[-1]) * t ** 2))
        loud, _ = delayandsum(self._cell(), source_waveform=wf, sample_rate=fs,
                              trace_frequency=self.FC)
        lossless = dict(self._cell(), delays_imag=np.array([0.0]))
        ref, _ = delayandsum(lossless, source_waveform=wf, sample_rate=fs,
                             trace_frequency=self.FC)
        f = np.fft.rfftfreq(loud.size, 1.0 / fs)
        ratio = np.abs(np.fft.rfft(loud)) / np.abs(np.fft.rfft(ref))
        for fk in (16000.0, 20000.0, 24000.0):
            k = int(np.argmin(np.abs(f - fk)))
            assert 20 * np.log10(ratio[k]) == pytest.approx(
                20 * np.log10(np.exp(2 * np.pi * f[k] * self.TAU_I)),
                abs=0.05)


class TestDelayAndSumSaysWhatTheWindowLeavesOut:
    """A delay-and-sum drops an echo the window does not hold — it never
    folds — so the record looks complete whatever it left out. The window
    report names the omitted and clipped echoes and the energy they carried."""

    FS = 100000.0
    FC = 10000.0

    def _cell(self, *delays):
        n = len(delays)
        return dict(n_arrivals=n,
                    amplitudes=np.ones(n), phases=np.zeros(n),
                    delays=np.asarray(delays, dtype=float),
                    delays_imag=np.zeros(n),
                    n_top_bounces=np.zeros(n, dtype=int),
                    n_bot_bounces=np.zeros(n, dtype=int),
                    source_angles=np.zeros(n), receiver_angles=np.zeros(n))

    def _waveform(self, n=200):
        # 2 ms hann-windowed tone burst
        k = np.arange(n)
        return np.hanning(n) * np.sin(2 * np.pi * self.FC * k / self.FS)

    def _run(self, cell, **kw):
        from uacpy.acoustic_signal import delayandsum
        return warning_messages(lambda: delayandsum(
            rcv_arrivals=cell, source_waveform=self._waveform(),
            sample_rate=self.FS, trace_frequency=self.FC, **kw),
            "does not hold every echo")

    def test_an_echo_beyond_the_window_is_counted_with_its_energy(self):
        # Two equal echoes; the window closes before the second begins, so
        # it is omitted and takes half the energy with it: -3 dB.
        msgs = self._run(self._cell(0.010, 0.050),
                         t_start=0.0, output_duration=0.030)
        assert len(msgs) == 1, msgs
        assert "1 echo(es) fall entirely outside" in msgs[0]
        assert "-3 dB of the received energy" in msgs[0]

    def test_a_window_holding_every_echo_says_nothing(self):
        assert self._run(self._cell(0.010, 0.020),
                         t_start=0.0, output_duration=0.040) == []

    def test_the_auto_window_holds_everything(self):
        assert self._run(self._cell(0.010, 0.050)) == []

    def test_a_cut_leading_edge_is_named_as_such(self):
        # The window opens 1 ms into the first echo's 2 ms waveform.
        msgs = self._run(self._cell(0.010, 0.020),
                         t_start=0.011, output_duration=0.040)
        assert len(msgs) == 1, msgs
        assert "1 echo(es) begin before it and lose their leading edge" in msgs[0]

    def test_an_echo_running_past_the_end_loses_its_tail(self):
        # The second echo starts 1 ms before a window that ends at 30 ms.
        msgs = self._run(self._cell(0.010, 0.029),
                         t_start=0.0, output_duration=0.030)
        assert len(msgs) == 1 and "run past its end" in msgs[0], msgs

    def test_the_notice_names_delayandsum_as_the_caller(self):
        msgs = self._run(self._cell(0.010, 0.050),
                         t_start=0.0, output_duration=0.030)
        assert msgs and msgs[0].startswith('delayandsum:'), msgs

    def test_a_report_dict_totals_the_cells_and_stays_silent(self):
        from uacpy.acoustic_signal import delayandsum
        report = {}
        with recorded_warnings() as rec:
            for cell in (self._cell(0.010, 0.050), self._cell(0.012, 0.060)):
                delayandsum(cell, source_waveform=self._waveform(),
                            sample_rate=self.FS, trace_frequency=self.FC,
                            t_start=0.0, output_duration=0.030, report=report)
        assert not [w for w in rec if "does not hold every echo"
                    in str(w.message)]
        assert report['omitted'] == 2
        assert report['omitted_power'] == pytest.approx(2.0)
        assert report['total_power'] == pytest.approx(4.0)


class TestTransferFunctionImpulseResponseValidatesItsSampleRate:
    """Every sibling entry point routes ``sample_rate`` through the shared
    positive-finite-scalar guard; this one did ``float(sample_rate)``.

    Measured before the fix, with ``frequencies`` spanning 0-500 Hz: a rate of
    0 raised a bare ``ZeroDivisionError``, NaN a ``ValueError`` about
    converting NaN to an integer, Inf an ``OverflowError``, a non-number a
    ``ValueError`` from ``float()`` — and a negative rate reached the Nyquist
    check and raised a ``ConfigurationError`` announcing "the Nyquist
    frequency -500 Hz", a true statement about the wrong argument.
    """

    @staticmethod
    def _call(rate):
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function,
        )
        f = np.arange(0.0, 501.0, 1.0)
        return impulse_response_from_transfer_function(
            np.ones(f.size, dtype=complex), frequencies=f, sample_rate=rate)

    @pytest.mark.parametrize('rate', [0.0, -1000.0, float('nan'),
                                      float('inf'), -float('inf')])
    def test_a_non_positive_or_non_finite_rate_names_sample_rate(self, rate):
        with pytest.raises(
                ConfigurationError,
                match='sample_rate must be > 0 Hz and finite') as exc:
            self._call(rate)
        message = str(exc.value)
        assert 'impulse_response_from_transfer_function' in message
        assert 'sample_rate' in message and '> 0 Hz' in message
        assert 'Nyquist' not in message, (
            f"a bad sample_rate must not be reported as a statement about the "
            f"frequency axis: {message}")

    def test_a_non_numeric_rate_names_sample_rate_too(self):
        with pytest.raises(ConfigurationError, match='sample_rate must be a '
                                                     'number, not text'):
            self._call('fast')

    def test_a_positive_finite_rate_returns_a_finite_response(self):
        t, h = self._call(2000.0)
        assert t.size == h.size > 0 and np.all(np.isfinite(h))


class TestSynthesizeTimeSeriesOnArrays:
    """``acoustic_signal.synthesize_time_series``: a waveform through
    ``H(f)`` on plain arrays, the density route ``Field.synthesize_time_series``
    runs."""

    FS = 8000.0

    def _setup(self, f0):
        from uacpy.acoustic_signal import tone_burst
        _, s = tone_burst(1000.0, 10, sample_rate=self.FS)
        f = np.arange(f0, 3000.0, 1.0)
        return s, f

    @pytest.mark.parametrize('f0', [25.0, 25.3])
    def test_a_delayed_arrival_lands_at_its_delay_with_its_amplitude(self, f0):
        """``H = A·exp(-2πifτ)`` returns ``A·s(t - τ)``: the continuous
        Hann-burst evaluated at ``t - τ`` matches to 1e-3, on a band that
        starts on a multiple of Δf and on one that does not."""
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(f0)
        n = s.size
        tau, amp = 0.5, 0.01
        t, p = synthesize_time_series(amp * np.exp(-2j * np.pi * f * tau),
                                      frequencies=f, source_waveform=s,
                                      sample_rate=self.FS, t_start=0.0)
        tq = t - tau
        w = np.where((tq >= 0) & (tq <= (n - 1) / self.FS),
                     0.5 - 0.5 * np.cos(2 * np.pi * tq * self.FS / (n - 1)),
                     0.0)
        ideal = amp * np.sin(2 * np.pi * 1000.0 * tq) * w
        assert np.linalg.norm(p - ideal) / np.linalg.norm(ideal) < 1e-3

    def test_it_is_the_core_the_field_method_runs(self):
        """The Field method adds a start time and a re-wrap, nothing to the
        trace: at the same ``t_start`` the two are bit-identical."""
        from uacpy.acoustic_signal import synthesize_time_series
        from uacpy.core.results import Field, SoundSpeeds
        s, f = self._setup(25.3)
        H = 0.01 * np.exp(-2j * np.pi * f * 0.5)
        field = Field(data=H[None, None, :],
                      coords={'depth': np.array([10.0]),
                              'range': np.array([750.0]), 'frequency': f},
                      speeds=SoundSpeeds(water_max=1500.0))
        trace = field.synthesize_time_series(source_waveform=s,
                                             sample_rate=self.FS, t_start=0.0)
        t, p = synthesize_time_series(H, frequencies=f, source_waveform=s,
                                      sample_rate=self.FS, t_start=0.0)
        assert np.array_equal(trace.coords['time'], t)
        assert np.array_equal(trace.data[0, 0], p)

    def test_the_frequency_axis_can_be_any_axis(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        H = np.exp(-2j * np.pi * f * 0.3)
        _, one = synthesize_time_series(H, frequencies=f, source_waveform=s,
                                        sample_rate=self.FS, t_start=0.0)
        _, last = synthesize_time_series(np.stack([H, 2 * H]), frequencies=f,
                                         source_waveform=s,
                                         sample_rate=self.FS, t_start=0.0)
        _, first = synthesize_time_series(np.stack([H, 2 * H]).T,
                                          frequencies=f, source_waveform=s,
                                          sample_rate=self.FS, axis=0,
                                          t_start=0.0)
        assert last.shape == (2, one.size) and first.shape == (one.size, 2)
        np.testing.assert_array_equal(last[1], 2 * one)
        np.testing.assert_array_equal(first[:, 0], one)

    def test_t_start_moves_the_record(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        H = np.exp(-2j * np.pi * f * 0.3)
        t, _ = synthesize_time_series(H, frequencies=f, source_waveform=s, sample_rate=self.FS, t_start=0.25)
        assert t[0] == 0.25

    @pytest.mark.parametrize('bad,match', [
        ('complex', 'must be a real pressure pulse'),
        ('2d', 'must be a 1-D signal')])
    def test_the_waveform_is_a_real_one_dimensional_signal(self, bad, match):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        wf = s.astype(complex) if bad == 'complex' else np.stack([s, s])
        with pytest.raises(ConfigurationError, match=match):
            synthesize_time_series(np.ones(f.size, complex), frequencies=f,
                                   source_waveform=wf, sample_rate=self.FS,
                                   t_start=0.0)

    def test_a_mismatched_axis_is_refused(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        with pytest.raises(ConfigurationError, match='frequencies has'):
            synthesize_time_series(np.ones(f.size - 1, complex), frequencies=f,
                                   source_waveform=s, sample_rate=self.FS,
                                   t_start=0.0)

    def test_a_nan_bin_makes_the_trace_nan_and_warns(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        H = np.ones((2, f.size), complex)
        H[1, 10] = np.nan
        with pytest.warns(UserWarning, match='1 of 2 cell'):
            _, p = synthesize_time_series(H, frequencies=f, source_waveform=s,
                                          sample_rate=self.FS, t_start=0.0)
        assert np.all(np.isfinite(p[0])) and np.all(np.isnan(p[1]))
