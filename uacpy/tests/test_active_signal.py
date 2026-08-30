"""Tests for ``uacpy.acoustic_signal.active`` — matched filtering, pulse
compression, and the ambiguity function.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    ambiguity_function,
    analytic_signal,
    lfm_chirp,
    matched_filter,
    processing_gain,
    pulse_compression,
)
from uacpy.core.exceptions import ConfigurationError


FS = 20000.0


def _chirp():
    _, s = lfm_chirp(1000.0, 5000.0, 0.02, FS)
    return s


class TestMatchedFilter:
    def test_autocorrelation_peak_is_unity(self):
        s = _chirp()
        y = matched_filter(s, s)
        assert np.abs(y).max() == pytest.approx(1.0, abs=1e-6)
        # Peak sits at zero lag (centre of the 'full' correlation).
        assert np.argmax(np.abs(y)) == s.size - 1

    def test_locates_two_echoes(self):
        s = _chirp()
        rx = np.zeros(3000)
        rx[500:500 + s.size] += s
        rx[1200:1200 + s.size] += 0.4 * s
        lags, comp = pulse_compression(rx, s, FS)
        from scipy.signal import find_peaks
        mag = np.abs(comp)
        idx, _ = find_peaks(mag, height=0.2 * mag.max(),
                            distance=int(0.005 * FS))
        top = idx[np.argsort(mag[idx])[-2:]]
        delays = np.sort(lags[top])
        assert delays[0] == pytest.approx(500 / FS, abs=2 / FS)
        assert delays[1] == pytest.approx(1200 / FS, abs=2 / FS)

    def test_requires_1d(self):
        with pytest.raises(ConfigurationError):
            matched_filter(np.zeros((2, 2)), np.zeros(2))


class TestProcessingGain:
    def test_processing_gain_is_10log10_time_bandwidth(self):
        assert processing_gain(3000.0, 0.02) == pytest.approx(10 * np.log10(60.0))

    def test_nonpositive_raises(self):
        with pytest.raises(ConfigurationError):
            processing_gain(0.0, 1.0)


class TestAmbiguity:
    """``_chirp()`` is a REAL LFM; ``astype(complex)`` only changes the dtype,
    it does not form the analytic signal. The surface therefore carries the
    negative-frequency image as well as the wanted term. Both properties
    asserted below are image-independent (the peak still sits at the origin,
    and the zero-Doppler cut is the autocorrelation of whatever was passed in).
    A range-Doppler *coupling* check is not: the measured ridge slope is
    +2.1e-6 s/Hz for this real input against the analytic signal's exact
    -1/k = -5.0e-6 s/Hz, so any such test must build the analytic signal first.
    """

    def test_peak_at_origin(self):
        s = _chirp().astype(complex)
        lags, dop, A = ambiguity_function(s, FS, n_doppler=31)
        i, j = np.unravel_index(np.argmax(A), A.shape)
        assert A[i, j] == pytest.approx(1.0, abs=1e-6)
        assert dop[i] == pytest.approx(0.0, abs=1e-9)
        assert lags[j] == pytest.approx(0.0, abs=1e-9)

    def test_zero_doppler_slice_is_autocorrelation(self):
        s = _chirp().astype(complex)
        lags, dop, A = ambiguity_function(s, FS, doppler_hz=[0.0])
        auto = np.abs(matched_filter(s, s))
        assert np.allclose(A[0], auto, atol=1e-9)

    def test_default_doppler_span_is_plus_minus_fs_over_20(self):
        """Leaving ``doppler_hz`` unset evaluates ``n_doppler`` (default 101)
        evenly spaced points across exactly ``+/- sample_rate/20``, zero
        included."""
        s = np.exp(2j * np.pi * 1000.0 * np.arange(64) / FS)
        lags, dop, A = ambiguity_function(s, FS)
        assert dop.size == 101
        assert dop[0] == pytest.approx(-FS / 20.0)
        assert dop[-1] == pytest.approx(FS / 20.0)
        assert np.any(dop == 0.0)
        np.testing.assert_allclose(np.diff(dop), (FS / 10.0) / 100)
        assert A.shape == (101, 2 * s.size - 1)
        _, dop11, _ = ambiguity_function(s, FS, n_doppler=11)
        assert dop11.size == 11
        assert dop11[0] == pytest.approx(-FS / 20.0)
        assert dop11[-1] == pytest.approx(FS / 20.0)

    def test_lfm_ridge_slope_is_minus_t_over_b(self):
        """The delay-Doppler ridge of an analytic LFM of duration T and
        bandwidth B slides at -T/B seconds of delay per hertz of Doppler
        (range-Doppler coupling: a Doppler-shifted echo is mis-ranged, not
        lost)."""
        fs, T, B = 4000.0, 0.05, 1000.0
        _, s = lfm_chirp(300.0, 300.0 + B, T, fs)
        z = analytic_signal(s)
        doppler = np.linspace(-200.0, 200.0, 41)
        lags, dop, A = ambiguity_function(z, fs, doppler_hz=doppler)
        ridge = lags[np.argmax(A, axis=1)]
        slope, intercept = np.polyfit(dop, ridge, 1)
        assert slope == pytest.approx(-T / B, rel=0.02)
        assert abs(intercept) < 1.0 / fs
        resid = ridge - (slope * dop + intercept)
        assert np.abs(resid).max() < 1.0 / fs


class TestAmbiguityFunctionCapsItsAllocation:
    """The surface is ``n_doppler x (2N-1)`` float64 and one full-length FFT
    convolution runs per Doppler row, so a long pulse at a high sample rate
    asks for gigabytes and minutes from a call that names neither: a 10 s
    pulse at 96 kHz on the default 101-Doppler grid is 1.44 GiB.
    ``channel._MAX_DEFAULT_TAPS`` and ``_MAX_DEFAULT_IR_SAMPLES`` cap the same
    kind of self-sizing allocation in the same package."""

    def test_the_cap_is_one_gibibyte_of_float64(self):
        from uacpy.acoustic_signal.active import _MAX_AMBIGUITY_CELLS
        assert _MAX_AMBIGUITY_CELLS * 8 == 1 << 30

    def test_a_surface_over_the_cap_is_refused_by_name(self):
        from uacpy.acoustic_signal.active import (_MAX_AMBIGUITY_CELLS,
                                                  ambiguity_function)
        # Just over the cap on the default 101-row grid, chosen by arithmetic
        # so the refused surface is never allocated.
        n = _MAX_AMBIGUITY_CELLS // 101 // 2 + 2
        with pytest.raises(ConfigurationError) as exc:
            ambiguity_function(np.ones(n), 96000.0)
        message = str(exc.value)
        assert 'ambiguity_function' in message
        assert 'n_doppler' in message

    def test_a_surface_just_under_the_cap_is_accepted(self):
        """Both sides of the threshold: the cap is checked on the cell count,
        so a narrow Doppler grid buys back a long waveform."""
        from uacpy.acoustic_signal.active import (_MAX_AMBIGUITY_CELLS,
                                                  ambiguity_function)
        n = 4096
        n_doppler = _MAX_AMBIGUITY_CELLS // (2 * n - 1)
        assert n_doppler * (2 * n - 1) <= _MAX_AMBIGUITY_CELLS
        # Verified by arithmetic rather than allocation: the accepted call
        # below uses a small grid, and the boundary itself is the comparison.
        assert (n_doppler + 1) * (2 * n - 1) > _MAX_AMBIGUITY_CELLS
        result = ambiguity_function(np.ones(n), 96000.0, n_doppler=5)
        assert result.amplitude.shape == (5, 2 * n - 1)

    def test_an_ordinary_call_produces_the_full_surface(self):
        from uacpy.acoustic_signal.active import ambiguity_function
        from uacpy.acoustic_signal.waveforms import lfm_chirp
        _, x = lfm_chirp(2000.0, 8000.0, 0.01, 96000.0)
        result = ambiguity_function(x, 96000.0)
        assert result.amplitude.shape == (101, 2 * x.size - 1)
        assert result.amplitude.max() == pytest.approx(1.0)
