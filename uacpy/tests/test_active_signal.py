"""Tests for ``uacpy.acoustic_signal.active`` — matched filtering, pulse
compression, and the ambiguity function.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    ambiguity_function,
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
    def test_value(self):
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
