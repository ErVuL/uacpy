"""Unit tests for the TIME_SERIES auto-derivation helpers.

Covers the harmonisation layer that makes ``run(run_mode=TIME_SERIES,
source_waveform=, sample_rate=, output_duration=)`` work uniformly
across RAM / Scooter / Kraken / Bellhop / OASP:

* ``PropagationModel._resolve_time_series_frequencies`` — derives a
  uniform frequency grid from the source-waveform spectrum when the
  caller doesn't pin one, and warns about what got picked.
* ``PropagationModel._pad_waveform_to_duration`` — zero-pads the
  waveform so ``Δf = 1/output_duration`` falls out of the synthesis.
* ``RAM._resolve_broadband_grid`` — derives the native (fc, Q, T) tuple
  from a multi-element frequency array, with user-pinned Q/T winning.
* ``output_duration=`` kwarg on the model wrappers — end-to-end check
  that the returned ``Field`` covers at least the requested duration.
* DFT-wraparound warning in ``Field.synthesize_time_series``.

The synthetic ``Field`` fixtures below all carry
``phase_reference=TRAVELLING_WAVE``, which is a precondition rather than
decoration: it declares that ``H(f)`` still carries the engineering
propagator ``exp(-i k0 r)``, so ``2*Re[ifft(H)]`` puts the causal arrival
at ``t = r/c0`` (``core/results/_base.py:37-41``). The synthesis helpers
branch on it, and a fixture tagged ``TIME_DOMAIN_NATIVE`` would exercise
a different code path.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError
from uacpy.models.base import RunMode
import uacpy


C_WATER = 1500.0
F_CENTER = 200.0
FS = 8000.0


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _gaussian_pulse(fc=F_CENTER, sigma=0.003, fs=FS, n_periods=8):
    duration = max(n_periods / fc, 6 * sigma)
    t = np.arange(0, duration, 1.0 / fs)
    tc = t - duration / 2
    return np.exp(-0.5 * (tc / sigma) ** 2) * np.cos(2 * np.pi * fc * tc)


def _make_env():
    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0,
        density=1.5, attenuation=0.5,
    )
    env = uacpy.Environment(
        name='pekeris', bathymetry=50.0, ssp=C_WATER, bottom=bottom,
    )
    source = uacpy.Source(depths=25.0, frequencies=F_CENTER)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 45, 5),
        ranges=np.linspace(20, 200, 8),
    )
    return env, source, receiver


# ─────────────────────────────────────────────────────────────────────────────
# _pad_waveform_to_duration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_binary  # constructs a model to reach its helper method
class TestPadWaveformToDuration:
    """Zero-padding helper used by every IFFT-based wrapper."""

    def setup_method(self):
        # Concrete subclass needed because PropagationModel is abstract.
        from uacpy.models import Scooter
        self.model = Scooter(verbose=False)

    def test_pads_short_waveform(self):
        wf = np.ones(100)
        out = self.model._pad_waveform_to_duration(wf, sample_rate=1000.0,
                                                   output_duration=1.0)
        assert len(out) == 1000
        # Pad is exactly zero, original samples preserved.
        assert np.array_equal(out[:100], wf)
        assert np.all(out[100:] == 0.0)

    def test_longer_waveform_passes_through(self):
        wf = np.ones(2000)
        out = self.model._pad_waveform_to_duration(wf, sample_rate=1000.0,
                                                   output_duration=1.0)
        assert out is wf  # no copy when no padding needed

    def test_none_output_duration_is_noop(self):
        wf = np.ones(100)
        out = self.model._pad_waveform_to_duration(wf, sample_rate=1000.0,
                                                   output_duration=None)
        assert out is wf

    def test_none_waveform_returns_none(self):
        out = self.model._pad_waveform_to_duration(None, sample_rate=1000.0,
                                                    output_duration=1.0)
        assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_time_series_frequencies
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_binary  # constructs a model to reach its helper method
class TestResolveTimeSeriesFrequencies:
    """Auto-derivation of the broadband freq grid from the waveform."""

    def setup_method(self):
        from uacpy.models import Scooter
        self.model = Scooter(verbose=False)
        self.source = uacpy.Source(depths=25.0, frequencies=F_CENTER)

    def test_non_time_series_passes_through(self):
        out = self.model._resolve_time_series_frequencies(
            RunMode.COHERENT_TL, None,
            source_waveform=_gaussian_pulse(), sample_rate=FS,
        )
        assert out is None

    def test_explicit_frequencies_bypasses_derivation(self):
        freqs_in = np.linspace(100, 300, 11)
        out = self.model._resolve_time_series_frequencies(
            RunMode.TIME_SERIES, freqs_in,
            source_waveform=_gaussian_pulse(), sample_rate=FS,
        )
        assert out is freqs_in  # user-supplied wins, no derivation

    def test_derives_from_waveform_spectrum_and_warns(self):
        wf = _gaussian_pulse()
        with pytest.warns(UserWarning, match=r"auto-derived"):
            freqs = self.model._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, None,
                source_waveform=wf, sample_rate=FS,
            )
        assert freqs is not None
        assert len(freqs) >= 2
        # Δf should equal sample_rate / n_samples (= 1/duration).
        df = float(np.mean(np.diff(freqs)))
        expected_df = FS / len(wf)
        assert df == pytest.approx(expected_df, rel=1e-6)
        # Centred near fc.
        f_centre = 0.5 * (freqs[0] + freqs[-1])
        assert abs(f_centre - F_CENTER) < F_CENTER * 0.3

    def test_raises_on_zero_waveform(self):
        wf = np.zeros(100)
        with pytest.raises(ConfigurationError, match='identically zero'):
            self.model._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, None,
                source_waveform=wf, sample_rate=FS,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RAM._resolve_broadband_grid
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_binary  # constructs a model to reach its helper method
class TestResolveBroadbandGrid:
    """RAM's (fc, Q, T) derivation from source.frequencies."""

    def setup_method(self):
        from uacpy.models import RAM
        self.RAM = RAM
        self.source_scalar = uacpy.Source(depths=25.0, frequencies=F_CENTER)

    def test_single_freq_uses_defaults(self):
        ram = self.RAM(verbose=False)
        fc, Q, T = ram._resolve_broadband_grid(self.source_scalar)
        assert fc == F_CENTER
        assert Q == 2.0  # broadband default
        assert T == 10.0  # broadband default

    def test_single_freq_respects_pinned_q_t(self):
        ram = self.RAM(verbose=False, Q=4.0, T=5.0)
        fc, Q, T = ram._resolve_broadband_grid(self.source_scalar)
        assert (fc, Q, T) == (F_CENTER, 4.0, 5.0)

    def test_multi_freq_auto_derives_and_warns(self):
        # Band [50, 350] Hz at Δf=0.5. The sweep is parameterised as
        # fc·[1 - 1/Q, 1 + 1/Q] with Δf = 1/T, so Q is fc over the band
        # HALF-width (models/ram.py:759, :795): 200/150 = 4/3, not 2/3.
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with pytest.warns(UserWarning, match=r"From the 601-element"):
            fc, Q, T = ram._resolve_broadband_grid(src)
        assert fc == pytest.approx(200.0)
        assert Q == pytest.approx(4.0 / 3.0, rel=1e-4)
        assert T == pytest.approx(2.0, rel=1e-4)

    def test_multi_freq_both_pinned_skips_warning(self):
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False, Q=1.333, T=2.0)
        # No UserWarning expected.
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            ram._resolve_broadband_grid(src)
        bb = [c for c in caught
              if 'BROADBAND' in str(c.message) and 'mpiramS' in str(c.message)]
        assert bb == []

    def test_non_uniform_spacing_raises(self):
        freqs = np.array([50.0, 60.0, 80.0, 200.0, 350.0])
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with pytest.raises(ConfigurationError, match='non-uniform'):
            ram._resolve_broadband_grid(src)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: output_duration on a real model run
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.requires_binary
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestOutputDurationEndToEnd:
    """A model run with ``output_duration=`` returns a Field whose time
    axis spans at least the requested window. Auto-derive warnings are
    expected here and filtered."""

    def test_scooter_respects_output_duration(self):
        from uacpy.models import Scooter
        env, source, receiver = _make_env()
        wf = _gaussian_pulse()  # ~40 ms long
        t_request = 0.15  # ask for much longer
        field = Scooter(verbose=False).run(
            env, source, receiver, run_mode=RunMode.TIME_SERIES,
            source_waveform=wf, sample_rate=FS, output_duration=t_request,
        )
        times = np.asarray(field.coords['time'])
        # Output covers at least t_request (within one sample).
        assert times[-1] - times[0] >= t_request - 1.0 / FS

    def test_ram_respects_output_duration(self):
        from uacpy.models import RAM
        env, source, receiver = _make_env()
        wf = _gaussian_pulse()
        t_request = 0.15
        field = RAM(verbose=False, dr=2.0, dz=1.0, c0=1500.0).run(
            env, source, receiver, run_mode=RunMode.TIME_SERIES,
            source_waveform=wf, sample_rate=FS, output_duration=t_request,
        )
        times = np.asarray(field.coords['time'])
        assert times[-1] - times[0] >= t_request - 1.0 / FS


# ─────────────────────────────────────────────────────────────────────────────
# DFT wraparound warning in synthesize_time_series
# ─────────────────────────────────────────────────────────────────────────────


class TestSynthesisCarriesMetadata:
    """Time-domain synthesis must carry the source Field's metadata forward
    (pinned-work-dir output paths per DOCUMENTATION §6, c0/c_min, …) —
    every other derived-Field path preserves it."""

    @staticmethod
    def _tf_with_metadata():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 300.0, 21)          # Δf = 10 Hz
        return Field(
            data=np.ones((1, 1, len(freqs)), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'grn_file': '/pinned/model.grn', 'c0': 1500.0},
        )

    def test_synthesize_time_series_keeps_source_metadata(self):
        tf = self._tf_with_metadata()
        wf = np.zeros(int(0.05 * FS))
        wf[: int(0.005 * FS)] = 1.0
        ts = tf.synthesize_time_series(source_waveform=wf, sample_rate=FS)
        assert ts.metadata['grn_file'] == '/pinned/model.grn'
        assert ts.metadata['c0'] == 1500.0
        assert ts.metadata['window'] == 'hann'          # synthesis keys too

    def test_to_time_trace_keeps_source_metadata(self):
        tf = self._tf_with_metadata()
        trace = tf.to_time_trace(depth=25.0, range=100.0)
        assert trace.metadata['grn_file'] == '/pinned/model.grn'
        assert trace.metadata['c0'] == 1500.0
        assert trace.metadata['window'] == 'hann'


class TestDFTWraparoundWarning:
    """``Field.synthesize_time_series`` should warn when the source
    waveform is longer than the IFFT period ``1/Δf``."""

    def test_warns_when_waveform_longer_than_dft_period(self):
        from uacpy.core.results import Field, PhaseReference

        # tf has Δf = 10 Hz → DFT period = 0.1 s.
        freqs = np.linspace(100.0, 300.0, 21)  # Δf = 10 Hz
        depths = np.array([25.0])
        ranges = np.array([100.0])
        H = np.ones((1, 1, len(freqs)), dtype=complex)
        tf = Field(
            data=H,
            coords={'depth': depths, 'range': ranges, 'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )
        # Waveform 0.2 s long > 0.1 s DFT period → expect warning.
        n_long = int(0.2 * FS)
        wf = np.zeros(n_long)
        wf[: int(0.005 * FS)] = 1.0  # short non-zero burst
        with pytest.warns(UserWarning, match=r"wraps back"):
            tf.synthesize_time_series(source_waveform=wf, sample_rate=FS)


class TestSynthesisRangeSpanWarning:
    """All cells share one time window anchored at the nearest cell; a
    receiver-range span wider than the window aliases far-range arrivals
    back into early bins. ``Field.synthesize_time_series`` must warn."""

    @staticmethod
    def _pure_delay_tf(ranges, df=25.0):
        from uacpy.core.results import Field, PhaseReference
        c0 = 1500.0
        freqs = np.arange(df, 16.0 * df + df, df)
        H = np.exp(-2j * np.pi * freqs[None, None, :]
                   * (np.asarray(ranges)[None, :, None] / c0))
        return Field(
            data=H,
            coords={'depth': np.array([50.0]),
                    'range': np.asarray(ranges, dtype=float),
                    'frequency': freqs},
            model='Synthetic', frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'c0': c0})

    def test_warns_when_range_span_exceeds_window(self):
        tf = self._pure_delay_tf([100.0, 3000.0])   # 1.93 s spread, ~1 s window
        wf = np.zeros(64); wf[0] = 1.0
        with pytest.warns(UserWarning, match=r"range span|wrap"):
            tf.synthesize_time_series(wf, sample_rate=4000.0)

    def test_no_span_warning_for_single_range(self):
        tf = self._pure_delay_tf([100.0])
        wf = np.zeros(64); wf[0] = 1.0
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            tf.synthesize_time_series(wf, sample_rate=4000.0)
        assert not [w for w in rec if 'range span' in str(w.message)]


class TestSynthesisSizeCap:
    """``Field.synthesize_time_series`` caps the *auto* IFFT length so a
    too-high sample_rate cannot silently allocate a multi-GB buffer / OOM;
    an explicit ``nfft=`` is the user's opt-in and bypasses the cap."""

    @staticmethod
    def _tf():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 300.0, 21)
        H = np.ones((1, 1, len(freqs)), dtype=complex)
        return Field(
            data=H,
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs, phase_reference=PhaseReference.TRAVELLING_WAVE)

    def _wf(self):
        wf = np.zeros(64); wf[:8] = 1.0
        return wf

    def test_huge_sample_rate_raises(self):
        with pytest.raises(ConfigurationError, match="safety cap"):
            self._tf().synthesize_time_series(
                source_waveform=self._wf(), sample_rate=1e9)

    def test_normal_sample_rate_ok(self):
        ts = self._tf().synthesize_time_series(
            source_waveform=self._wf(), sample_rate=1e4)
        assert ts.kind == 'pressure' and 'time' in ts.coords
        assert ts.data.shape[-1] <= (1 << 26)

    def test_explicit_nfft_bypasses_cap(self):
        ts = self._tf().synthesize_time_series(
            source_waveform=self._wf(), sample_rate=1e9, nfft=4096)
        assert ts.data.shape[-1] == 4096


class TestSynthesisAbsoluteAmplitude:
    """A flat ``H ≡ 1`` must reproduce the source waveform's amplitude,
    independent of ``nfft`` (Fourier synthesis is a Riemann sum of the
    continuous inverse transform, not a raw bin-count-scaled IFFT)."""

    @staticmethod
    def _flat_tf():
        from uacpy.core.results import Field, PhaseReference

        freqs = np.arange(1.0, 301.0, 1.0)
        H = np.ones((1, 1, freqs.size), dtype=complex)
        return Field(
            data=H,
            coords={'depth': np.array([50.0]), 'range': np.array([1000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'c0': C_WATER},
        )

    @pytest.mark.parametrize('nfft', [None, 2048, 4096, 8192])
    def test_flat_h_reproduces_unit_peak(self, nfft):
        fs = 2000.0
        t = np.arange(2000) / fs
        src = (np.exp(-0.5 * ((t - 0.5) / 0.05) ** 2)
               * np.cos(2 * np.pi * 100.0 * (t - 0.5)))
        ts = self._flat_tf().synthesize_time_series(
            src, fs, window='none', nfft=nfft, t_start=0.0,
        )
        peak = float(np.abs(ts.data).max())
        # A gate, not a precision budget: with H ≡ 1 the synthesis returns
        # the source peak to machine precision (~4e-15 here). What the
        # ``nfft`` parametrisation catches is a bin-count-dependent scaling,
        # which would spread the peak over the 4x span of the nfft values —
        # far outside 2%.
        assert peak == pytest.approx(1.0, rel=0.02)

    def test_impulse_response_grid_independent(self):
        tf = self._flat_tf()
        a = tf.to_time_trace(window='none', nfft=4096, t_start=0.0)
        b = tf.to_time_trace(window='none', nfft=8192, t_start=0.0)
        assert float(np.abs(a.data).max()) == pytest.approx(
            float(np.abs(b.data).max()), rel=1e-6,
        )


class TestSourceSpectrumAtArbitraryFrequencies:
    """``_source_spectrum_at`` must be exact off the waveform's own DFT grid.

    Linear interpolation of ``rfft(w)/fs`` is not exact: it is a convolution
    with a triangular kernel in frequency — a ``sinc^2(pi df_src t)`` taper
    anchored at t=0, plus periodisation at ``1/df_src``. It agrees with the
    DTFT only where the target grid coincides with the source grid, so the
    off-grid cases below are the ones that can tell the two apart.
    """

    @staticmethod
    def _wf(n=256, fs=2000.0):
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * 100.0 * t) * np.hanning(n), fs

    @staticmethod
    def _dtft(wf, fs, freqs):
        n = wf.size
        return (wf[None, :] * np.exp(
            -2j * np.pi * np.asarray(freqs)[:, None] * np.arange(n)[None, :] / fs
        )).sum(1) / fs

    def test_matches_rfft_on_the_native_grid(self):
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._wf()
        grid = np.fft.rfftfreq(wf.size, 1.0 / fs)
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), np.fft.rfft(wf) / fs,
            rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize('shift', [0.5, 0.25])
    def test_exact_on_a_half_bin_offset_grid(self, shift):
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._wf()
        native = np.fft.rfftfreq(wf.size, 1.0 / fs)
        grid = native[:-1] + shift * (native[1] - native[0])
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), self._dtft(wf, fs, grid),
            rtol=1e-9, atol=1e-12)

    def test_exact_on_a_finer_grid(self):
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._wf()
        grid = np.fft.rfftfreq(4 * wf.size, 1.0 / fs)
        grid = grid[grid <= fs / 2]
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), self._dtft(wf, fs, grid),
            rtol=1e-9, atol=1e-12)

    def test_out_of_band_frequencies_are_zero(self):
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._wf()
        out = _source_spectrum_at(wf, fs, np.array([-10.0, fs, 2 * fs]))
        assert np.all(out == 0)

    def test_chunking_does_not_change_the_result(self):
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._wf()
        grid = np.linspace(10.0, 900.0, 137)
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid, _max_elems=1000),
            _source_spectrum_at(wf, fs, grid),
            rtol=1e-12, atol=1e-15)


class TestNarrowBandWindowDoesNotAnnihilate:
    """np.hanning(2) == [0, 0] and np.hanning(3) == [0, 1, 0], so tapering the
    *frequency* axis of a 2- or 3-bin band returns silence or a pure tone. The
    taper exists to soften the band edges; with no interior left there is
    nothing to soften."""

    @staticmethod
    def _tf(n_freq):
        from uacpy.core.results import Field
        return Field(data=np.ones((1, 1, n_freq), dtype=complex),
                     coords={'depth': [100.0], 'range': [3000.0],
                             'frequency': np.linspace(450.0, 550.0, n_freq)})

    def _trace(self, n_freq):
        from uacpy.core.results.field import _ifft_to_trace
        return np.asarray(_ifft_to_trace(
            self._tf(n_freq), depth=100.0, range=3000.0,
            source_spectrum=np.ones(n_freq, dtype=complex),
            window='hann', nfft=None, t_start=0.0).data)

    @pytest.mark.parametrize('n_freq', [2, 3])
    def test_degenerate_band_is_not_zeroed(self, n_freq):
        with pytest.warns(UserWarning, match="too narrow to taper"):
            y = self._trace(n_freq)
        assert np.abs(y).max() > 0.0, "the whole band was multiplied by zero"

    def test_wide_band_is_unaffected(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error')
            y = self._trace(32)
        assert np.abs(y).max() > 0.0


@pytest.mark.requires_binary
def test_auto_derived_timeseries_grid_resolves_the_band():
    """A 20 ms burst gives Delta f = 50 Hz, so a 450-550 Hz band derives only
    3 bins — which the frequency-axis taper then collapses to a CW tone. The
    derived grid must carry enough bins to represent an arrival."""
    import warnings as _w
    from scipy.signal import hilbert
    from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties,
                       Source, Receiver, Kraken)
    from uacpy.models.base import RunMode

    env = Environment(
        bathymetry=200.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5))
    fs, dur = 20000.0, 0.020
    t = np.arange(0.0, dur, 1.0 / fs)
    wf = np.hanning(t.size) * np.sin(2 * np.pi * 500.0 * t)
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        r = Kraken(timeout=300).run(
            env, Source(depths=50.0, frequencies=500.0),
            Receiver(depths=[100.0], ranges=[3000.0]),
            run_mode=RunMode.TIME_SERIES, source_waveform=wf, sample_rate=fs)
    y = np.real(np.asarray(r.data)).ravel()
    envelope = np.abs(hilbert(y))
    tt = np.arange(y.size) / fs
    core = envelope[(tt > 0.05) & (tt < 0.95)]
    assert core.min() / core.max() < 0.5, (
        "envelope is flat — the band collapsed to a CW tone rather than an "
        "impulse response")


class TestNoSincSquaredTaperOnTheFieldSpectrum:
    """Refining df below the transfer function's own spacing has to invent the
    samples in between, and linear interpolation of the spectrum is a
    triangular kernel — a sinc^2(pi df_data t) taper that eats arrivals away
    from its anchor. Two arrivals of known ratio must return at that ratio
    whatever the frequency spacing."""

    @staticmethod
    def _two_arrival_trace(df_data, dtau, a2=0.5):
        from uacpy.core.results import Field
        from uacpy.core.results.field import _ifft_to_trace
        freqs = np.arange(50.0, 450.0 + df_data, df_data)
        # exp(-2i pi f tau) is a unit impulse at t = tau.
        H = (np.exp(-2j * np.pi * freqs * 0.010)
             + a2 * np.exp(-2j * np.pi * freqs * (0.010 + dtau)))
        tf = Field(data=H[None, None, :],
                   coords={'depth': [50.0], 'range': [1000.0],
                           'frequency': freqs})
        tr = _ifft_to_trace(
            tf, depth=50.0, range=1000.0,
            source_spectrum=np.ones(freqs.size, dtype=complex),
            window='none', nfft=None, t_start=0.0)
        return (np.asarray(tr.coords['time']),
                np.abs(np.asarray(tr.data)).ravel())

    @pytest.mark.parametrize('dtau', [0.020, 0.040, 0.060])
    def test_arrival_ratio_survives_a_coarse_grid(self, dtau):
        t, y = self._two_arrival_trace(10.0, dtau)
        dt = float(t[1] - t[0])
        w = max(1, int(0.002 / dt))

        def peak_near(tau):
            i = int(np.argmin(np.abs(t - tau)))
            return y[max(0, i - w):i + w + 1].max()

        ratio = peak_near(0.010 + dtau) / peak_near(0.010)
        assert ratio == pytest.approx(0.5, abs=0.1), (
            f"second arrival returned at {ratio:.3f} of the first instead of "
            f"0.5 — a sinc^2 taper is attenuating it with separation")

    def test_record_length_matches_the_grid_it_came_from(self):
        """1/df_data is the non-aliased extent; anything longer is fabricated."""
        t, _ = self._two_arrival_trace(10.0, 0.020)
        assert (t[-1] - t[0]) <= 1.0 / 10.0 + 2 * float(t[1] - t[0])


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-grid bin alignment in _ifft_to_trace
# ─────────────────────────────────────────────────────────────────────────────


class TestSynthesisBinAlignment:
    """A DFT of spacing Δf carries only integer multiples of Δf.

    When ``f[0]`` is not itself a multiple of Δf the whole band is placed at
    an offset of up to Δf/2, which frequency-shifts the trace. The synthesis
    removes that offset, so a grid the caller happened to build with
    ``linspace`` reconstructs as well as one built with ``arange``.
    """

    FS = 8000.0
    BAND = (280.0, 360.0)

    def _source(self):
        n = 400
        t = np.arange(n) / self.FS
        wf = np.sin(2 * np.pi * 320.0 * t) * np.hanning(n)
        spec = np.fft.rfft(wf)
        f = np.fft.rfftfreq(n, 1.0 / self.FS)
        spec[(f < self.BAND[0]) | (f > self.BAND[1])] = 0.0
        return t, np.fft.irfft(spec, n)

    def _flat_channel_error(self, freqs):
        """Max error of ``H == 1`` synthesis against the source waveform."""
        t, wf = self._source()
        tf = uacpy.Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': [10.0], 'range': [0.0], 'frequency': freqs},
            model='X', frequencies=freqs, phase_reference='travelling_wave')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = tf.synthesize_time_series(wf, self.FS, t_start=0.0,
                                            window='none')
        y = out.data[0, 0, :]
        ref = np.interp(out.coords['time'], t, wf, left=0.0, right=0.0)
        return float(np.max(np.abs(y[:ref.size] - ref)) / np.max(np.abs(wf)))

    @staticmethod
    def _bin_offset(freqs):
        df = float(freqs[1] - freqs[0])
        return float(np.floor(freqs[0] / df + 0.5) * df - freqs[0])

    def test_aligned_grid_reproduces_the_source(self):
        freqs = np.arange(260.0, 380.1, 10.0)
        assert self._bin_offset(freqs) == pytest.approx(0.0, abs=1e-9)
        assert self._flat_channel_error(freqs) < 0.05

    @pytest.mark.parametrize('freqs', [
        np.linspace(261.0, 381.0, 13),    # offset -1 Hz
        np.linspace(255.0, 375.0, 13),    # offset +5 Hz
        np.linspace(266.0, 386.0, 9),     # offset +4 Hz, coarser df
    ])
    def test_misaligned_grid_reproduces_the_source(self, freqs):
        """Without the de-rotation the band lands shifted by the bin offset."""
        assert abs(self._bin_offset(freqs)) > 0.5
        assert self._flat_channel_error(freqs) < 0.05

    def test_auto_derived_timeseries_grid_keeps_the_carrier(self):
        """``_MIN_TIMESERIES_FREQS`` refinement produces a misaligned grid.

        The refined ``linspace(f_min, f_max, 8)`` is exactly the case a
        default TIME_SERIES run derives, and its band is narrower than the
        source, so the assertion is on the carrier rather than on waveform
        equality: the trace must sit at the frequency the caller asked for,
        not at that frequency minus the bin offset.
        """
        model = uacpy.models.Bellhop.__new__(uacpy.models.Bellhop)
        model.model_name = 'Bellhop'
        t, wf = self._source()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            freqs = model._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, None, wf, self.FS)
        assert freqs.size >= 8
        assert abs(self._bin_offset(freqs)) > 0.5, "grid is already aligned"

        tf = uacpy.Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': [10.0], 'range': [0.0], 'frequency': freqs},
            model='X', frequencies=freqs, phase_reference='travelling_wave')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = tf.synthesize_time_series(wf, self.FS, t_start=0.0,
                                            window='none')
        y = out.data[0, 0, :]
        dt = float(out.coords['time'][1] - out.coords['time'][0])
        spec = np.abs(np.fft.rfft(y, 16384))
        peak = float(np.fft.rfftfreq(16384, dt)[np.argmax(spec)])
        assert peak == pytest.approx(320.0, abs=1.0)


class TestSynthesisWindowAnchor:
    """The output window must open before the earliest arrival.

    The earliest arrival travels at the fastest speed, so the anchor is
    ``r / c_max``; anchoring on the slowest speed opens the window after it.
    """

    FREQS = np.arange(40.0, 81.0, 1.0)      # 1 Hz spacing -> 1 s record

    def _trace(self, metadata, range_m=60000.0):
        tf = uacpy.Field(
            data=np.ones((1, 1, self.FREQS.size), dtype=complex),
            coords={'depth': [100.0], 'range': [range_m],
                    'frequency': self.FREQS},
            model='RAM', frequencies=self.FREQS,
            phase_reference='travelling_wave', metadata=metadata)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            trace = tf.to_time_trace()
        return float(trace.coords['time'][0]), caught

    def test_c_max_anchors_before_the_earliest_arrival(self):
        t0, caught = self._trace({'c_min': 1500.0, 'c_max': 1550.0,
                                  'c0': 1520.0})
        assert t0 <= 60000.0 / 1550.0
        assert not [w for w in caught if 'wrap to the end' in str(w.message)]

    def test_missing_c_max_warns_at_long_range(self):
        _, caught = self._trace({'c_min': 1500.0, 'c0': 1520.0})
        assert [w for w in caught if 'wrap to the end' in str(w.message)], (
            "a long-range trace with no c_max must say the window start is "
            "an estimate")

    def test_short_range_does_not_warn(self):
        t0, caught = self._trace({'c0': 1500.0}, range_m=2000.0)
        assert t0 <= 2000.0 / 1500.0
        assert not [w for w in caught if 'wrap to the end' in str(w.message)]
