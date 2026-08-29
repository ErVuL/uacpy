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
at ``t = r/c0`` (``PhaseReference.TRAVELLING_WAVE``). The synthesis helpers
branch on it, and a fixture tagged ``TIME_DOMAIN_NATIVE`` would exercise
a different code path.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError
from uacpy import Source
from uacpy.models.base import RunMode
from uacpy.models.bellhop import Bellhop
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

    def test_single_freq_collapses_to_one_bin(self):
        # A single frequency with neither Q nor T pinned asks for a 1-bin
        # H(f): the sweep collapses the same way COHERENT_TL does, and the
        # requested-bins helper trims the result to exactly that bin.
        ram = self.RAM(verbose=False)
        fc, Q, T = ram._resolve_broadband_grid(self.source_scalar)
        assert fc == F_CENTER
        assert Q == 1e6
        assert T == 1.0
        target = ram._requested_broadband_bins(self.source_scalar)
        assert list(target) == [F_CENTER]

    def test_single_freq_respects_pinned_q_t(self):
        ram = self.RAM(verbose=False, Q=4.0, T=5.0)
        fc, Q, T = ram._resolve_broadband_grid(self.source_scalar)
        assert (fc, Q, T) == (F_CENTER, 4.0, 5.0)

    def test_multi_freq_auto_derives_and_warns(self):
        # Band [50, 350] Hz at Δf=0.5. fc anchors on the upper-middle array
        # bin and Q = fc / ((n//2 + 1/2)·Δf), so the marched (fc, Q, T)
        # sweep reproduces every requested bin — the property that matters —
        # rather than a nominal fc/half-width ratio.
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with pytest.warns(UserWarning, match=r"From the 601-element"):
            fc, Q, T = ram._resolve_broadband_grid(src)
        assert fc == pytest.approx(200.0)
        assert T == pytest.approx(2.0, rel=1e-4)
        marched = ram._broadband_frequencies(fc, Q, T)
        assert marched.size == freqs.size
        assert np.allclose(marched, freqs)

    def test_multi_freq_with_both_pinned_warns_and_names_the_sweep(self):
        # A frequency array and a pinned (Q, T) pair each define the sweep;
        # the pins win (compute_time_series legitimately derives an array
        # while the user pins the sweep), but the replacement used to be
        # silent — now the warning names both grids.
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False, Q=1.333, T=2.0)
        with pytest.warns(UserWarning, match="pinned"):
            fc, q, t = ram._resolve_broadband_grid(src)
        assert fc == pytest.approx(200.0)    # the middle array bin (odd count)
        assert (q, t) == (1.333, 2.0)

    def test_non_uniform_spacing_raises(self):
        freqs = np.array([50.0, 60.0, 80.0, 200.0, 350.0])
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with pytest.raises(ConfigurationError, match='non-uniform'):
            ram._resolve_broadband_grid(src)

    def test_single_freq_partial_pin_fills_in_broadband_defaults(self):
        """ram.md §5: with exactly one of Q/T pinned, the other fills in at
        the broadband defaults Q=2.0 / T=10.0 (not the narrowband 1e6 / 1.0
        collapse), and the warning names which value was defaulted."""
        with pytest.warns(UserWarning, match='default'):
            fc, Q, T = self.RAM(verbose=False, T=5.0)._resolve_broadband_grid(
                self.source_scalar)
        assert (fc, Q, T) == (F_CENTER, 2.0, 5.0)
        with pytest.warns(UserWarning, match='default'):
            fc, Q, T = self.RAM(verbose=False, Q=8.0)._resolve_broadband_grid(
                self.source_scalar)
        assert (fc, Q, T) == (F_CENTER, 8.0, 10.0)


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


class TestSynthesisCarriesPinned:
    """Time-domain synthesis inherits the parent Field's pinned axes (the
    accumulation contract in the Field class doc); ``to_time_trace`` adds
    the synthesised cell's coordinates on top."""

    @staticmethod
    def _tf_pinned():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 300.0, 21)          # Δf = 10 Hz
        return Field(
            data=np.ones((1, 1, len(freqs)), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            pinned={'source_depth': 5.0},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

    def test_to_time_trace_merges_pinned_under_cell_coords(self):
        trace = self._tf_pinned().to_time_trace(depth=25.0, range=100.0)
        assert trace.pinned['source_depth'] == 5.0
        assert trace.pinned['depth'] == 25.0
        assert trace.pinned['range'] == 100.0

    def test_synthesize_time_series_keeps_pinned(self):
        wf = np.zeros(int(0.05 * FS))
        wf[: int(0.005 * FS)] = 1.0
        ts = self._tf_pinned().synthesize_time_series(
            source_waveform=wf, sample_rate=FS)
        assert ts.pinned == {'source_depth': 5.0}


class TestToTimeTraceDefaultCell:
    """results.md §6: with no arguments ``to_time_trace`` takes the middle
    depth and the first range, recording the chosen cell in ``pinned``."""

    @staticmethod
    def _tf():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 300.0, 21)          # Δf = 10 Hz
        depths = np.array([10.0, 20.0, 30.0])
        ranges = np.array([500.0, 1000.0])
        # Amplitude encodes the cell so the defaulted pick is observable in
        # the trace, not only in the pinned labels.
        amp = 1.0 + np.arange(3)[:, None] * 10.0 + np.arange(2)[None, :]
        data = amp[:, :, None] * np.ones((1, 1, freqs.size), dtype=complex)
        return Field(
            data=data,
            coords={'depth': depths, 'range': ranges, 'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

    def test_defaults_to_middle_depth_first_range(self):
        tf = self._tf()
        trace = tf.to_time_trace()
        assert trace.pinned['depth'] == 20.0
        assert trace.pinned['range'] == 500.0
        # The (20 m, 500 m) cell carries |H| = 11 against 2 at
        # (10 m, 1000 m); the same synthesis on both cells preserves that
        # amplitude ratio, so the defaulted pick is visible in the data too.
        other = tf.to_time_trace(depth=10.0, range=1000.0)
        ratio = (float(np.max(np.abs(trace.data)))
                 / float(np.max(np.abs(other.data))))
        assert ratio == pytest.approx(11.0 / 2.0, rel=1e-6)

    def test_explicit_cell_wins(self):
        trace = self._tf().to_time_trace(depth=30.0, range=1000.0)
        assert trace.pinned['depth'] == 30.0
        assert trace.pinned['range'] == 1000.0


class TestSynthesisPhaseReferenceContract:
    """results.md §6: both synthesis methods refuse a
    ``'time_domain_native'`` input (that payload is already p(t)) and tag
    their own output ``'time_domain_native'``."""

    @staticmethod
    def _tf(phase_reference):
        from uacpy.core.results import Field
        freqs = np.linspace(100.0, 300.0, 21)
        return Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs,
            phase_reference=phase_reference,
        )

    def test_native_input_is_refused_by_both_entry_points(self):
        from uacpy.core.results import PhaseReference
        tf = self._tf(PhaseReference.TIME_DOMAIN_NATIVE)
        wf = np.zeros(int(0.05 * FS))
        wf[: int(0.005 * FS)] = 1.0
        with pytest.raises(ConfigurationError, match='time_domain_native'):
            tf.to_time_trace(depth=25.0, range=100.0)
        with pytest.raises(ConfigurationError, match='time_domain_native'):
            tf.synthesize_time_series(source_waveform=wf, sample_rate=FS)

    def test_output_is_tagged_time_domain_native(self):
        from uacpy.core.results import PhaseReference
        tf = self._tf(PhaseReference.TRAVELLING_WAVE)
        wf = np.zeros(int(0.05 * FS))
        wf[: int(0.005 * FS)] = 1.0
        trace = tf.to_time_trace(depth=25.0, range=100.0)
        series = tf.synthesize_time_series(source_waveform=wf, sample_rate=FS)
        assert trace.phase_reference == 'time_domain_native'
        assert series.phase_reference == 'time_domain_native'


class TestManualIfftRecipe:
    """DOCUMENTATION.md §'Manual IFFT': the documented zero-padded-buffer
    recipe, executed verbatim on the phase-only ``H = exp(-i 2π f r/c0)``
    the doc names (r = 3000 m, c0 = 1500 m/s) — the impulse must land at
    exactly t = 2.0 s."""

    def test_impulse_lands_at_two_seconds(self):
        from uacpy.core.results import Field, PhaseReference
        r, c0 = 3000.0, 1500.0
        freqs = np.arange(50.0, 400.0 + 0.125, 0.25)   # 1/Δf = 4 s > r/c0
        H = Field(
            data=np.exp(-2j * np.pi * freqs * r / c0)[None, None, :],
            coords={'depth': np.array([50.0]), 'range': np.array([r]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([50.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

        # The doc recipe, verbatim.
        f = H.coords['frequency']
        spec1d = H.at(depth=50, range=3000).data
        df = f[1] - f[0]
        nfft = 1 << int(np.ceil(np.log2(2 * round(f[-1] / df) + 2)))
        buf = np.zeros(nfft, complex)
        buf[np.round(f / df).astype(int)] = spec1d
        pt = 2.0 * np.real(np.fft.ifft(buf)) * (nfft * df)
        t = np.arange(nfft) / (nfft * df)

        peak = float(t[np.argmax(pt)])
        assert peak == pytest.approx(r / c0, abs=2.0 / (nfft * df)), (
            f"impulse at {peak:.4f} s, expected {r / c0:.4f} s")
        # A real impulse, not a ripple: the peak dominates the record.
        assert float(np.max(pt)) > 10.0 * float(np.median(np.abs(pt)))


class TestSynthesisErrorsNameTheEntryPoint:
    """``_synthesis_plan`` diagnostics carry the public entry point's name,
    so the message names the method the caller actually invoked."""

    @staticmethod
    def _tf_one_freq():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.array([200.0])
        return Field(
            data=np.ones((1, 1, 1), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

    def test_to_time_trace_label(self):
        with pytest.raises(ConfigurationError,
                           match="to_time_trace: need at least 2"):
            self._tf_one_freq().to_time_trace(depth=25.0, range=100.0)

    def test_synthesize_time_series_label(self):
        wf = np.zeros(64)
        wf[:8] = 1.0
        with pytest.raises(ConfigurationError,
                           match="synthesize_time_series: need at least 2"):
            self._tf_one_freq().synthesize_time_series(
                source_waveform=wf, sample_rate=FS)

    @staticmethod
    def _tf_nan_cell_no_stamped_speed():
        """A Field that trips both shared synthesis warnings at once.

        The second range cell is entirely NaN, and the metadata carries no
        ``c_max``/``c0``, so ``_clean_cell_spectra`` and the window-anchor
        branch of ``_ifft_to_trace`` both fire on either entry point."""
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)
        data = np.ones((1, 2, freqs.size), dtype=complex)
        data[0, 1, :] = np.nan
        return Field(
            data=data,
            coords={'depth': np.array([20.0]),
                    'range': np.array([1000.0, 2000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

    def test_to_time_trace_warnings_carry_its_label(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            self._tf_nan_cell_no_stamped_speed().to_time_trace(
                depth=20.0, range=2000.0)
        messages = [str(w.message) for w in record]
        assert any('entirely NaN' in m for m in messages), messages
        assert any('stamped no sound speed' in m for m in messages), messages
        assert all(m.startswith('to_time_trace: ') for m in messages), messages

    def test_synthesize_time_series_warnings_carry_its_label(self):
        wf = np.zeros(64)
        wf[:8] = 1.0
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            self._tf_nan_cell_no_stamped_speed().synthesize_time_series(
                source_waveform=wf, sample_rate=4000.0)
        messages = [str(w.message) for w in record]
        assert any('entirely NaN' in m for m in messages), messages
        assert any('stamped no sound speed' in m for m in messages), messages
        assert all(m.startswith('synthesize_time_series: ') for m in messages), \
            messages

    @staticmethod
    def _tf_c0_only_short_window():
        """A Field whose window-start estimate carries the fast/slow spread.

        ``c0`` is stamped and ``c_max`` is not, and 5 % of the 66.7 s travel
        time exceeds the 0.01 s of lead a 0.02 s window offers, which is the
        third shared diagnostic in ``_ifft_to_trace``."""
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)          # Δf = 50 Hz
        return Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': np.array([20.0]),
                    'range': np.array([100000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'c0': C_WATER},
        )

    def test_short_window_warning_carries_the_entry_points_label(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            self._tf_c0_only_short_window().to_time_trace(
                depth=20.0, range=100000.0)
        messages = [str(w.message) for w in record]
        assert any('synthesis window is short' in m for m in messages), messages
        assert all(m.startswith('to_time_trace: ') for m in messages), messages


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


def _outer_product_dtft(wf, fs, freqs):
    """The dense evaluation the chirp-z path stands in for, kept as reference.

    Same sum, same out-of-band rule, one frequency per row of an explicit
    phase matrix.
    """
    wf = np.asarray(wf, dtype=np.float64).ravel()
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    out = np.zeros(freqs.size, dtype=np.complex128)
    sel = np.flatnonzero((freqs >= 0.0) & (freqs <= 0.5 * fs))
    if sel.size:
        phase = np.exp(-2j * np.pi * np.outer(
            freqs[sel], np.arange(wf.size, dtype=np.float64)) / fs)
        out[sel] = phase @ wf
    return out / fs


def _rel_norm(ref, got):
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(np.asarray(ref) - np.asarray(got)) / denom
                 ) if denom else float(np.linalg.norm(np.asarray(ref) - got))


class TestSourceSpectrumChirpZEqualsTheOuterProduct:
    """A uniform ascending frequency grid is a chirp-z contour — with
    ``z_k = a*w**-k``, ``a = exp(2i*pi*f0/fs)`` and ``w = exp(-2i*pi*df/fs)``,
    scipy's ``czt`` sums exactly the DTFT ``_source_spectrum_at`` documents,
    by FFT convolution instead of an (n_freq x n_sample) phase matrix.

    Two things then need pinning, and neither is the speed. The transform has
    to return what the dense sum returns, on the ordinary grids and on the
    awkward ones (a single bin, a grid crossing or clearing Nyquist, negative
    frequencies, a one- or two-sample waveform). And it must NOT be
    unconditional: the contour passes through the requested frequencies only
    where they are uniformly spaced, while the function's contract is
    arbitrary ones — a caller passing ``[-10, fs, 2*fs]`` is in this same
    file.
    """

    CASES = [
        # (label, n_wf, fs, grid)
        ('native rfft grid', 256, 2000.0, np.fft.rfftfreq(256, 1 / 2000.0)),
        ('half-bin offset', 256, 2000.0,
         np.fft.rfftfreq(256, 1 / 2000.0)[:-1] + 2000.0 / 512),
        ('finer than native', 256, 2000.0,
         np.linspace(0.0, 1000.0, 1024)),
        ('narrow in-band band', 400, 8000.0, np.linspace(50.0, 2000.0, 512)),
        ('single bin in band', 512, 1000.0, np.array([100.0])),
        ('single bin at zero', 512, 1000.0, np.array([0.0])),
        ('single bin out of band', 512, 1000.0, np.array([9e9])),
        ('one-sample waveform', 1, 1000.0, np.linspace(0.0, 500.0, 33)),
        ('two-sample waveform', 2, 1000.0, np.linspace(0.0, 500.0, 9)),
        ('grid straddles nyquist', 128, 1000.0, np.linspace(0.0, 900.0, 19)),
        ('grid clears nyquist', 128, 1000.0, np.linspace(600.0, 900.0, 7)),
        ('grid reaches below zero', 128, 1000.0,
         np.linspace(-200.0, 400.0, 13)),
    ]

    @staticmethod
    def _waveform(n, seed=17):
        return np.random.default_rng(seed).standard_normal(n)

    @pytest.mark.parametrize('label,n_wf,fs,grid', CASES,
                             ids=[c[0] for c in CASES])
    def test_it_matches_the_outer_product(self, label, n_wf, fs, grid):
        from uacpy.core.results.field import _source_spectrum_at
        wf = self._waveform(n_wf)
        ref = _outer_product_dtft(wf, fs, grid)
        got = _source_spectrum_at(wf, fs, grid)
        assert got.shape == ref.shape
        assert _rel_norm(ref, got) < 1e-9
        # The out-of-band zeros are exact zeros on both routes, not small.
        np.testing.assert_array_equal(got == 0, ref == 0)

    def test_an_all_zero_waveform_returns_exact_zeros(self):
        from uacpy.core.results.field import _source_spectrum_at
        got = _source_spectrum_at(np.zeros(64), 1000.0,
                                  np.linspace(0.0, 400.0, 11))
        assert np.array_equal(got, np.zeros(11, dtype=np.complex128))

    def test_a_non_uniform_grid_keeps_the_dense_sum(self):
        from uacpy.core.results.field import _source_spectrum_at, _chirp_step
        wf, fs = self._waveform(256), 2000.0
        grid = np.geomspace(20.0, 900.0, 64)          # ascending, not uniform
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid),
            _outer_product_dtft(wf, fs, grid), rtol=1e-13, atol=1e-16)
        assert _chirp_step(grid, wf.size, fs) is None

    def test_the_dense_fallback_chunks_over_frequency(self):
        # ``_max_elems`` bounds the phase matrix, and only the dense path
        # builds one now — so the block arithmetic is exercised on a grid the
        # contour cannot serve rather than on the uniform one above, where
        # both calls would take the chirp-z route and agree vacuously.
        from uacpy.core.results.field import _source_spectrum_at
        wf, fs = self._waveform(256), 2000.0
        grid = np.geomspace(20.0, 900.0, 137)
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid, _max_elems=1000),
            _source_spectrum_at(wf, fs, grid), rtol=1e-13, atol=1e-16)

    def test_the_chirp_contour_would_be_wrong_on_that_grid(self):
        # Why the fallback is not decoration: the contour is anchored at f[0]
        # and steps by a constant, so on a non-uniform grid it evaluates the
        # spectrum at frequencies nobody asked for.
        from scipy.signal import czt
        wf, fs = self._waveform(256), 2000.0
        grid = np.geomspace(20.0, 900.0, 64)
        df = (grid[-1] - grid[0]) / (grid.size - 1)
        contour = czt(wf, m=grid.size, w=np.exp(-2j * np.pi * df / fs),
                      a=np.exp(2j * np.pi * grid[0] / fs)) / fs
        assert _rel_norm(_outer_product_dtft(wf, fs, grid), contour) > 0.1

    def test_a_grid_uniform_only_to_a_loose_tolerance_is_refused(self):
        # The contour has to LAND on the frequencies, not merely resemble
        # them: a drift that a spacing-ratio test would wave through is a
        # phase error growing with waveform length.
        from uacpy.core.results.field import _chirp_step
        grid = np.linspace(100.0, 900.0, 401)
        grid[200] += 1e-4                       # 1e-7 of the span
        assert _chirp_step(grid, 4096, 8000.0) is None
        assert _chirp_step(np.linspace(100.0, 900.0, 401), 4096, 8000.0) \
            == pytest.approx(2.0)

    def test_the_synthesised_trace_matches_the_dense_sum(self, monkeypatch):
        # End to end through the public entry point, against the same Field
        # synthesised with the dense sum forced back in.
        import uacpy.core.results.field as F
        from uacpy.core.results import Field, PhaseReference
        rng = np.random.default_rng(4)
        freqs = np.linspace(50.0, 2000.0, 256)
        data = ((rng.standard_normal((2, 3, freqs.size)) +
                 1j * rng.standard_normal((2, 3, freqs.size)))
                / (1.0 + np.arange(freqs.size)))
        tf = Field(data=data,
                   coords={'depth': np.array([10.0, 20.0]),
                           'range': np.array([100.0, 200.0, 300.0]),
                           'frequency': freqs},
                   model='Synthetic', source_depths=np.array([5.0]),
                   frequencies=freqs,
                   phase_reference=PhaseReference.TRAVELLING_WAVE)
        wf = np.hanning(400) * np.sin(
            2 * np.pi * 700 * np.arange(400) / 8000.0)
        got = tf.synthesize_time_series(source_waveform=wf, sample_rate=8000.0)
        monkeypatch.setattr(F, '_chirp_step', lambda *a, **k: None)
        ref = tf.synthesize_time_series(source_waveform=wf, sample_rate=8000.0)
        a, b = np.asarray(ref.data, float), np.asarray(got.data, float)
        assert np.abs(a - b).max() / np.abs(a).max() < 1e-12
        assert np.array_equal(ref.coords['time'], got.coords['time'])


def _never_called(*args, **kwargs):
    raise AssertionError("_source_spectrum_at ran before the axis was checked")


class TestSynthesisChecksTheFrequencyAxisBeforeUsingIt:
    """``_synthesis_plan`` refuses a non-uniform frequency axis, but the
    synthesis evaluates the source spectrum on that axis before it ever builds
    a plan — and the chirp-z evaluation assumes the same uniform grid the plan
    demands. So the refusal has to come first, or a broken axis gets a
    computed answer before it gets its error.
    """

    @staticmethod
    def _tf(freqs):
        from uacpy.core.results import Field, PhaseReference
        return Field(
            data=np.ones((1, 1, len(freqs)), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': np.asarray(freqs, dtype=float)},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=np.asarray(freqs, dtype=float),
            phase_reference=PhaseReference.TRAVELLING_WAVE)

    def test_a_non_uniform_axis_raises_before_the_spectrum_is_evaluated(
            self, monkeypatch):
        import uacpy.core.results.field as F
        monkeypatch.setattr(F, '_source_spectrum_at', _never_called)
        tf = self._tf([100.0, 110.0, 130.0, 140.0])
        with pytest.raises(ConfigurationError, match='uniformly spaced'):
            tf.synthesize_time_series(source_waveform=np.ones(64),
                                      sample_rate=FS)

    def test_a_descending_axis_is_refused_the_same_way(self, monkeypatch):
        import uacpy.core.results.field as F
        monkeypatch.setattr(F, '_source_spectrum_at', _never_called)
        with pytest.raises(ConfigurationError, match='uniformly spaced'):
            self._tf([300.0, 200.0, 100.0]).synthesize_time_series(
                source_waveform=np.ones(64), sample_rate=FS)

    def test_a_uniform_axis_synthesises(self):
        ts = self._tf(np.linspace(100.0, 300.0, 21)).synthesize_time_series(
            source_waveform=np.ones(64), sample_rate=FS)
        assert ts.n_times > 0


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

    def test_misaligned_refined_grid_keeps_the_carrier(self):
        """A refinement-misaligned grid must still synthesise the carrier.

        ``_MIN_TIMESERIES_FREQS`` refinement subdivides the waveform Δf, so a
        refined grid can sit between the record's own FFT bins. (The default
        9-bin refinement of this source happens to land exactly on-grid, so
        the misaligned 8-bin variant is built explicitly.) The band is
        narrower than the source, so the assertion is on the carrier rather
        than on waveform equality: the trace must sit at the frequency the
        caller asked for, not at that frequency minus the bin offset.
        """
        model = uacpy.models.Bellhop.__new__(uacpy.models.Bellhop)
        model.model_name = 'Bellhop'
        t, wf = self._source()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            derived = model._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, None, wf, self.FS)
        freqs = np.linspace(derived[0], derived[-1], 8)
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

    def test_c_min_never_binds_the_anchor(self):
        # Only fastest-speed candidates may anchor the window: r/c_min is an
        # upper bound on the arrival, and anchoring on it opens the window
        # early enough that the true arrival wraps to the end of the record.
        t0, _ = self._trace({'c_min': 5000.0, 'c_max': 1550.0})
        assert t0 == pytest.approx(60000.0 / 1550.0 - 0.5, abs=1e-9)

    def test_pe_reference_speed_never_binds_the_anchor(self):
        # RAM stamps its Padé expansion point as 'pe_reference_speed'; it is
        # an algorithmic constant, often above every physical speed, so it
        # must not enter the physical-speed max.
        t0, _ = self._trace({'pe_reference_speed': 1700.0, 'c_max': 1550.0})
        assert t0 == pytest.approx(60000.0 / 1550.0 - 0.5, abs=1e-9)


class TestAllNaNCellPropagatesNaN:
    """An all-NaN H(f) cell (e.g. one masked below the seafloor) carries no
    valid model output, so its synthesised trace is NaN with a warning —
    never a silent all-zero record that reads as a real quiet arrival.
    Isolated NaN bins still count as carrying no energy (zeroed)."""

    @staticmethod
    def _tf(H):
        from uacpy.core.results import Field, PhaseReference
        n_d, n_r, n_f = H.shape
        freqs = np.arange(50.0, 50.0 + 2.0 * n_f, 2.0)
        return Field(
            data=H,
            coords={'depth': np.linspace(10.0, 90.0, n_d),
                    'range': np.linspace(500.0, 3000.0, n_r),
                    'frequency': freqs},
            model='Synthetic', frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'c0': 1500.0, 'c_max': 1520.0})

    def test_to_time_trace_warns_and_returns_nan(self):
        tf = self._tf(np.full((1, 1, 16), np.nan, dtype=complex))
        with pytest.warns(UserWarning, match='entirely NaN'):
            trace = tf.to_time_trace()
        assert np.all(np.isnan(trace.data))

    def test_isolated_nan_bins_stay_no_energy(self):
        rng = np.random.default_rng(0)
        H = (rng.standard_normal((1, 1, 16))
             + 1j * rng.standard_normal((1, 1, 16)))
        H[0, 0, 3] = np.nan
        trace = self._tf(H).to_time_trace()
        assert np.all(np.isfinite(trace.data))
        assert np.any(trace.data != 0.0)

    def test_synthesize_keeps_valid_cells_and_nans_the_dead_one(self):
        rng = np.random.default_rng(1)
        H = (rng.standard_normal((2, 2, 16))
             + 1j * rng.standard_normal((2, 2, 16)))
        H[1, 0, :] = np.nan
        wf = np.zeros(32); wf[0] = 1.0
        with pytest.warns(UserWarning, match='entirely NaN'):
            out = self._tf(H).synthesize_time_series(wf, sample_rate=500.0)
        assert np.all(np.isnan(out.data[1, 0]))
        for di, ri in ((0, 0), (0, 1), (1, 1)):
            assert np.all(np.isfinite(out.data[di, ri]))


class TestBatchedSynthesisMatchesPerCellTraces:
    """``synthesize_time_series`` computes every cell through batched iffts;
    each cell of the grid must reproduce ``to_time_trace`` at that cell run
    with the same shared window."""

    def test_grid_equals_per_cell_traces(self):
        from uacpy.core.results import Field, PhaseReference
        from uacpy.core.results.field import _ifft_to_trace, _source_spectrum_at
        rng = np.random.default_rng(2)
        freqs = np.arange(40.0, 40.0 + 2.0 * 32, 2.0)
        depths = np.linspace(10.0, 40.0, 2)
        ranges = np.linspace(1000.0, 4000.0, 3)
        H = (rng.standard_normal((2, 3, 32))
             + 1j * rng.standard_normal((2, 3, 32)))
        tf = Field(
            data=H, coords={'depth': depths, 'range': ranges,
                            'frequency': freqs},
            model='Synthetic', frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            metadata={'c0': 1500.0, 'c_max': 1520.0})
        fs = 500.0
        wf = _gaussian_pulse(fc=70.0, fs=fs)
        out = tf.synthesize_time_series(wf, sample_rate=fs)
        t_start = float(out.coords['time'][0])
        nfft = out.coords['time'].size
        src = _source_spectrum_at(np.asarray(wf, float), fs, freqs)
        for di in range(depths.size):
            for ri in range(ranges.size):
                tr = _ifft_to_trace(
                    tf, depth=float(depths[di]), range=float(ranges[ri]),
                    source_spectrum=src, window='hann', nfft=nfft,
                    t_start=t_start)
                np.testing.assert_allclose(
                    out.data[di, ri], tr.data, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(out.data)) > 0.0


def _waveform():
    return np.sin(2.0 * np.pi * 100.0 * np.arange(64) / 1000.0)


@pytest.mark.requires_binary
class TestTimeSeriesGuardReturnsARealFloatWaveform:
    """``_require_timeseries_signal`` admits a complex waveform whose
    imaginary part is ~0 on purpose, and every downstream consumer casts
    with ``dtype=float`` — a cast that raises a bare ``TypeError`` on a
    complex Python list and emits ``ComplexWarning`` on a complex ndarray.
    The guard therefore returns the waveform to run with: the float64 real
    part for accepted complex input, the caller's object otherwise."""

    @pytest.mark.parametrize('kind', ['list', 'ndarray'])
    def test_an_accepted_complex_waveform_comes_back_as_float64_real(
            self, kind):
        wf = [complex(v, 0.0) for v in _waveform()]
        if kind == 'ndarray':
            wf = np.asarray(wf)
        m = Bellhop(verbose=False)
        ret = m._require_timeseries_signal(RunMode.TIME_SERIES, wf, 1000.0)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.float64
        assert np.array_equal(ret, _waveform())

    @pytest.mark.parametrize('kind', ['list', 'ndarray'])
    def test_downstream_casts_of_the_returned_waveform_stay_silent(
            self, kind):
        wf = [complex(v, 0.0) for v in _waveform()]
        if kind == 'ndarray':
            wf = np.asarray(wf)
        m = Bellhop(verbose=False)
        ret = m._require_timeseries_signal(RunMode.TIME_SERIES, wf, 1000.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            padded = m._pad_waveform_to_duration(ret, 1000.0, 0.1)
            freqs = m._resolve_time_series_frequencies(
                RunMode.TIME_SERIES, np.array([50.0, 100.0]), ret, 1000.0)
        assert np.asarray(padded).dtype == np.float64
        assert freqs is not None

    def test_prepare_timeseries_hands_downstream_the_coerced_waveform(self):
        m = Bellhop(verbose=False)
        wf, freqs = m._prepare_timeseries(
            RunMode.TIME_SERIES, Source(depths=25.0, frequencies=100.0),
            np.array([50.0, 100.0]), [complex(v, 0.0) for v in _waveform()],
            1000.0)
        assert np.asarray(wf).dtype == np.float64

    @pytest.mark.parametrize('kind', ['list', 'ndarray'])
    def test_a_significant_imaginary_part_is_refused(self, kind):
        wf = [v + 0.5j for v in _waveform()]
        if kind == 'ndarray':
            wf = np.asarray(wf)
        with pytest.raises(ConfigurationError, match='imaginary'):
            Bellhop(verbose=False)._require_timeseries_signal(
                RunMode.TIME_SERIES, wf, 1000.0)

    def test_a_real_waveform_passes_through_as_the_same_object(self):
        wf = _waveform()
        ret = Bellhop(verbose=False)._require_timeseries_signal(
            RunMode.TIME_SERIES, wf, 1000.0)
        assert ret is wf

    def test_broadband_with_no_waveform_returns_none(self):
        ret = Bellhop(verbose=False)._require_timeseries_signal(
            RunMode.BROADBAND, None, None)
        assert ret is None
