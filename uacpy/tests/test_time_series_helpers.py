"""Unit tests for the TIME_SERIES auto-derivation helpers.

Covers the harmonisation layer that makes ``run(run_mode=TIME_SERIES,
source_waveform=, sample_rate=, output_duration=)`` work uniformly
across RAM / Scooter / KrakenField / Bellhop / OASP:

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
"""

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


class TestResolveTimeSeriesFrequencies:
    """Auto-derivation of the broadband freq grid from the waveform."""

    def setup_method(self):
        from uacpy.models import Scooter
        self.model = Scooter(verbose=False)
        self.source = uacpy.Source(depths=25.0, frequencies=F_CENTER)

    def test_non_time_series_passes_through(self):
        out = self.model._resolve_time_series_frequencies(
            RunMode.COHERENT_TL, self.source, None,
            source_waveform=_gaussian_pulse(), sample_rate=FS,
        )
        assert out is None

    def test_explicit_frequencies_bypasses_derivation(self):
        freqs_in = np.linspace(100, 300, 11)
        out = self.model._resolve_time_series_frequencies(
            RunMode.TIME_SERIES, self.source, freqs_in,
            source_waveform=_gaussian_pulse(), sample_rate=FS,
        )
        assert out is freqs_in  # user-supplied wins, no derivation

    def test_derives_from_waveform_spectrum_and_warns(self):
        wf = _gaussian_pulse()
        with pytest.warns(UserWarning, match=r"auto-derived"):
            freqs = self.model._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, self.source, None,
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
                RunMode.TIME_SERIES, self.source, None,
                source_waveform=wf, sample_rate=FS,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RAM._resolve_broadband_grid
# ─────────────────────────────────────────────────────────────────────────────


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
        # Band [50, 350] Hz at Δf=0.5 → fc=200, Q=4/3, T=2.0
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
