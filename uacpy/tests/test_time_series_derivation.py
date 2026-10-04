"""What a TIME_SERIES run derives when the caller does not pin it.

Covers the harmonisation layer that makes ``run(run_mode=TIME_SERIES,
source_waveform=, sample_rate=, output_duration=)`` work uniformly
across RAM / Scooter / Kraken / Bellhop / OASP:

* ``uacpy.models._band.time_series_band`` — derives a uniform frequency
  grid from the source-waveform spectrum when the caller doesn't pin one,
  with the notice a run gives about what got picked.
* ``uacpy.models._band.pad_waveform_to_duration`` — zero-pads the
  waveform so ``Δf = 1/output_duration`` falls out of the synthesis.
* ``ram._band.resolve_broadband_grid`` — derives the native (fc, Q, T) tuple
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
from uacpy.core.results import SoundSpeeds
from uacpy.core.run_settings import RunMode
from uacpy.models.bellhop import Bellhop
import uacpy
from uacpy.tests.conftest import recorded_warnings


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
# pad_waveform_to_duration
# ─────────────────────────────────────────────────────────────────────────────


class TestPadWaveformToDuration:
    """Zero-padding helper used by every IFFT-based wrapper."""

    def setup_method(self):
        from uacpy.models._band import pad_waveform_to_duration
        self.pad = pad_waveform_to_duration

    def test_pads_short_waveform(self):
        wf = np.ones(100)
        out = self.pad(wf, sample_rate=1000.0, output_duration=1.0)
        assert len(out) == 1000
        # Pad is exactly zero, original samples preserved.
        assert np.array_equal(out[:100], wf)
        assert np.all(out[100:] == 0.0)

    def test_longer_waveform_passes_through(self):
        wf = np.ones(2000)
        out = self.pad(wf, sample_rate=1000.0, output_duration=1.0)
        assert out is wf  # no copy when no padding needed

    def test_none_output_duration_is_noop(self):
        wf = np.ones(100)
        out = self.pad(wf, sample_rate=1000.0, output_duration=None)
        assert out is wf

    def test_none_waveform_returns_none(self):
        out = self.pad(None, sample_rate=1000.0, output_duration=1.0)
        assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_band / time_series_band
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveTimeSeriesFrequencies:
    """Auto-derivation of the broadband freq grid from the waveform."""

    def setup_method(self):
        from uacpy.core.run_settings import TimeSettings
        self.source = uacpy.Source(depths=25.0, frequencies=F_CENTER)
        self.time = TimeSettings(source_waveform=_gaussian_pulse(),
                                 sample_rate=FS, output_duration=None,
                                 t_start=None)

    def test_a_single_frequency_mode_reads_the_source(self):
        from uacpy.models._band import resolve_band
        band = resolve_band(RunMode.COHERENT_TL, self.source, None,
                            self.time, model_name='Scooter')
        np.testing.assert_array_equal(band.frequencies, [F_CENTER])
        assert band.notice is None

    def test_explicit_frequencies_bypasses_derivation(self):
        from uacpy.models._band import resolve_band
        freqs_in = np.linspace(100, 300, 11)
        band = resolve_band(RunMode.TIME_SERIES, self.source, freqs_in,
                            self.time, model_name='Scooter')
        assert band.frequencies is freqs_in  # user-supplied wins
        assert band.notice is None

    def test_a_multi_element_source_band_is_the_grid(self):
        from uacpy.models._band import resolve_band
        source = uacpy.Source(depths=25.0,
                              frequencies=np.linspace(150., 250., 11))
        band = resolve_band(RunMode.TIME_SERIES, source, None, self.time,
                            model_name='Scooter')
        np.testing.assert_array_equal(band.frequencies, source.frequencies)
        assert band.notice is None

    def test_a_one_element_source_band_derives_from_the_pulse(self):
        from uacpy.models._band import resolve_band
        band = resolve_band(RunMode.TIME_SERIES, self.source, None,
                            self.time, model_name='Scooter')
        assert band.frequencies.size > 1
        assert 'auto-derived' in band.notice

    @pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
    @pytest.mark.parametrize('source_freqs, expected', [
        (100.0, (10.0, 280.0, 28)),
        (np.arange(60.0, 140.5, 0.5), (60.0, 140.0, 161)),
    ])
    def test_a_model_time_series_run_marches_the_source_band(
            self, source_freqs, expected):
        from uacpy.acoustic_signal import lfm_chirp
        from uacpy.models import Kraken
        _, pulse = lfm_chirp(80.0, 120.0, 0.1, sample_rate=1000.0)
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
        source = uacpy.Source(depths=36.0, frequencies=source_freqs)
        receiver = uacpy.Receiver(depths=50.0, ranges=3000.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = Kraken().run_settings(
                env, source, receiver, run_mode=RunMode.TIME_SERIES,
                source_waveform=pulse, sample_rate=1000.0)
        freqs = settings.frequencies
        assert (freqs[0], freqs[-1], freqs.size) == expected

    def test_derives_from_waveform_spectrum_with_a_notice(self):
        from uacpy.models._band import time_series_band
        wf = _gaussian_pulse()
        band = time_series_band(wf, FS, model_name='Scooter')
        assert 'auto-derived' in band.notice
        freqs = band.frequencies
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
        from uacpy.models._band import time_series_band
        wf = np.zeros(100)
        with pytest.raises(ConfigurationError, match='identically zero'):
            time_series_band(wf, FS, model_name='Scooter')


def _burst(n_cycles, f0=80.0, fs=2000.0):
    """A rectangular tone burst of ``n_cycles`` cycles."""
    n = int(round(n_cycles / f0 * fs))
    return np.sin(2.0 * np.pi * f0 * np.arange(n) / fs), fs


class TestTheRecordOpensJustBeforeTheFirstArrival:
    """``record_start``: a tenth of the record (or four inverse bandwidths,
    whichever is longer, at most half) before ``r / c`` on the fastest
    stamped speed, and a warning when the record ends before the
    arrival at the slowest stated water speed."""

    def test_a_tenth_of_the_record_leads_the_estimated_arrival(self):
        from uacpy.acoustic_signal._synthesis import (
            RECORD_LEAD_FRACTION, record_start)
        assert RECORD_LEAD_FRACTION == 0.1
        # 1 s over 200 Hz: the onset term 4/200 = 0.02 s is under a tenth.
        t0 = record_start(3000.0, 1.0, bandwidth=200.0, c_max=1700.0,
                          c0=0.0, c_slow=0.0, who='t')
        assert t0 == pytest.approx(3000.0 / 1700.0 - 0.1, abs=1e-12)

    @pytest.mark.parametrize('bandwidth, lead', [
        (40.0, 0.1), (39.0, 4.0 / 39.0), (4.0, 0.5)])
    def test_a_narrow_band_widens_the_lead_to_hold_its_onset(self,
                                                             bandwidth, lead):
        from uacpy.acoustic_signal._synthesis import record_lead
        assert record_lead(1.0, bandwidth) == pytest.approx(lead, abs=1e-12)

    @pytest.mark.parametrize('c_slow, warns', [(1224.0, True),
                                               (1225.0, False)])
    def test_a_record_ending_before_the_slow_arrival_warns(self, c_slow,
                                                           warns):
        from uacpy.acoustic_signal._synthesis import record_start
        # [1.95, 2.45] s at 3 km: r/c_slow = 2.4510 s at 1224 m/s, 2.4490
        # at 1225 m/s.
        with recorded_warnings() as record:
            t0 = record_start(3000.0, 0.5, bandwidth=1000.0, c_max=1500.0,
                              c0=0.0, c_slow=c_slow, who='t')
        assert t0 == pytest.approx(1.95, abs=1e-12)
        messages = [str(w.message) for w in record
                    if 'folds onto' in str(w.message)]
        assert len(messages) == (1 if warns else 0)


class TestTheBandIsMeasuredOffTheRecordLattice:
    """RA-WAVE-4: the band of a pulse is measured on the pulse zero-padded
    well past its length, then placed on the record's bins, and a pulse
    with a hard edge is limited to one -20 dB band-width beyond its -20 dB
    band (``log/fix-r2-models_scratch/b1_band_rules.out``)."""

    @staticmethod
    def _grid(wf, fs):
        from uacpy.models._band import _time_series_grid
        return _time_series_grid(wf, fs, -40.0)

    def test_an_integer_cycle_burst_keeps_its_main_lobe(self):
        """Unpadded, a 4.0-cycle burst's DFT is one line at 80 Hz, and the
        band was [80, 100] Hz; its main lobe spans 60-100 Hz."""
        grid = self._grid(*_burst(4.0))
        assert grid.freq_min <= 60.0 and grid.freq_max >= 100.0

    def test_a_pulse_length_record_keeps_a_tapered_tone_skirts(self):
        """A 40-sample Hann-tapered 100 Hz tone at 400 Hz: the record's 10 Hz
        lattice sits on its spectral nulls, and the band was [90, 110] Hz
        (8.8 % of its energy outside, -0.80 dB). Its -40 dB support is
        71.4-128.6 Hz; the record bins inside it are 80-120 Hz, and the band
        takes one more on each side, 70-130 Hz, where the pulse is below
        -40 dB."""
        t = np.arange(40) / 400.0
        grid = self._grid(np.hanning(40) * np.sin(2 * np.pi * 100.0 * t),
                          400.0)
        assert (grid.freq_min, grid.freq_max) == (70.0, 130.0)
        assert grid.energy_outside < 1e-3

    @pytest.mark.parametrize('padding', [1, 4, 20])
    def test_a_smooth_pulse_takes_the_record_bins_of_its_support(
            self, padding):
        """A Gaussian burst sampled finely enough by the record gets the
        record bins above -40 dB on the record's own DFT and one more bin
        on each side, and no limit."""
        wf = _gaussian_pulse()
        wf = np.concatenate([wf, np.zeros((padding - 1) * wf.size)])
        grid = self._grid(wf, FS)
        spectrum = np.abs(np.fft.rfft(wf))
        bins = np.fft.rfftfreq(wf.size, 1.0 / FS)
        df = FS / wf.size
        above = bins[spectrum >= spectrum.max() * 1e-2]
        assert (grid.freq_min, grid.freq_max) == (max(above[0] - df, df),
                                            above[-1] + df)
        assert not grid.limited

    def test_a_hard_edged_burst_stops_one_core_width_beyond_its_core(self):
        """A 4.2-cycle burst ends on a jump; its -40 dB support reaches
        990 Hz. The band's top is the last record bin at or below the -20 dB
        band's top plus that band's width, measured independently here on
        the 16x-padded spectrum."""
        wf, fs = _burst(4.2)
        grid = self._grid(wf, fs)
        m = 16
        spectrum = np.abs(np.fft.rfft(wf, m * wf.size))
        core = np.flatnonzero(spectrum >= spectrum.max() * 0.1)
        top = core[-1] + (core[-1] - core[0])
        df = fs / wf.size
        assert grid.freq_max == pytest.approx((top // m) * df)
        assert grid.support_max > 900.0
        assert grid.limited and grid.freq_max_origin.startswith('the -20 dB')
        assert 0.005 < grid.energy_outside < 0.02

    @pytest.mark.parametrize('padding', [1, 4])
    def test_the_band_edges_stand_below_the_ring_threshold(self, padding):
        """The synthesis's untapered band-edge check (-40 dB re the band's
        peak) stays silent on the auto band: each edge is one record bin past
        the -40 dB support, while the last bin inside it is still above."""
        from uacpy.acoustic_signal._synthesis import _BAND_EDGE_RING_DB
        wf = _gaussian_pulse()
        wf = np.concatenate([wf, np.zeros((padding - 1) * wf.size)])
        grid = self._grid(wf, FS)
        n = np.arange(wf.size)

        def level(f):
            return np.abs(np.exp(-2j * np.pi * np.outer(f, n) / FS) @ wf)

        f = grid.frequencies
        band = level(f)
        edge_dB = 20 * np.log10(max(band[0], band[-1]) / band.max())
        assert edge_dB <= _BAND_EDGE_RING_DB
        df = FS / wf.size
        inner = level(np.array([grid.freq_min + df, grid.freq_max - df]))
        assert 20 * np.log10(inner.max() / band.max()) > _BAND_EDGE_RING_DB

    def test_a_chosen_record_drops_the_record_notice(self):
        """With ``output_duration=`` the record is the caller's choice, so
        the notice that the pulse set it is silent; without, it speaks."""
        from uacpy.models._band import time_series_band
        wf = _gaussian_pulse()
        assert time_series_band(wf, FS, model_name='Bellhop',
                                record_chosen=True).notice is None
        assert 'output_duration' in time_series_band(
            wf, FS, model_name='Bellhop').notice

    def test_a_chosen_record_keeps_the_limit_notice(self):
        from uacpy.models._band import time_series_band
        wf, fs = _burst(4.2)
        said = time_series_band(wf, fs, model_name='Scooter',
                                record_chosen=True).notice
        assert 'falls slowly' in said and 'record' not in said

    def test_resolve_band_reads_the_chosen_record_off_the_time_settings(self):
        from uacpy.core.run_settings import TimeSettings
        from uacpy.models._band import resolve_band
        source = uacpy.Source(depths=25.0, frequencies=F_CENTER)
        wf = _gaussian_pulse()
        for duration, silent in ((None, False), (0.2, True)):
            time = TimeSettings(source_waveform=wf, sample_rate=FS,
                                output_duration=duration)
            notice = resolve_band(uacpy.RunMode.TIME_SERIES, source, None,
                                  time, model_name='Bellhop').notice
            assert (notice is None) is silent

    def test_the_notice_states_the_limit_and_the_full_band_cost(self):
        from uacpy.models._band import time_series_band
        wf, fs = _burst(4.2)
        band = time_series_band(wf, fs, model_name='Scooter')
        said = band.notice
        assert 'falls slowly' in said
        assert f"instead of {band.frequencies.size}" in said
        assert '% of the pulse' in said


# ─────────────────────────────────────────────────────────────────────────────
# ram._band.resolve_broadband_grid
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
        fc, Q, T = ram_band.resolve_broadband_grid(self.source_scalar,
                                                   knobs=ram._knob_record(),
                                                   log=ram._log)
        assert fc == F_CENTER
        assert Q == 1e6
        assert T == 1.0
        target = ram_band.requested_broadband_bins(self.source_scalar,
                                                   knobs=ram._knob_record())
        assert list(target) == [F_CENTER]

    def test_single_freq_respects_pinned_q_t(self):
        ram = self.RAM(verbose=False, q_factor=4.0, record_duration=5.0)
        fc, Q, T = ram_band.resolve_broadband_grid(self.source_scalar,
                                                   knobs=ram._knob_record(),
                                                   log=ram._log)
        assert (fc, Q, T) == (F_CENTER, 4.0, 5.0)

    def test_multi_freq_auto_derives_silently(self):
        # Band [50, 350] Hz at Δf=0.5. fc anchors on the upper-middle array
        # bin and Q = fc / ((n//2 + 1/2)·Δf), so the marched (fc, Q, T)
        # sweep reproduces every requested bin — the property that matters —
        # rather than a nominal fc/half-width ratio. The result carries
        # exactly the requested bins, so the derivation warns nothing.
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fc, Q, T = ram_band.resolve_broadband_grid(
                src, knobs=ram._knob_record(), log=ram._log)
        assert fc == pytest.approx(200.0)
        assert T == pytest.approx(2.0, rel=1e-4)
        marched = ram_band.broadband_frequencies(fc, Q, T)
        assert marched.size == freqs.size
        assert np.allclose(marched, freqs)

    def test_multi_freq_with_both_pinned_warns_and_names_the_sweep(self):
        # A frequency array and a pinned (Q, T) pair each define the sweep;
        # the pins win (compute_time_series legitimately derives an array
        # while the user pins the sweep), but the replacement used to be
        # silent — now the warning names both grids.
        freqs = np.linspace(50.0, 350.0, 601)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False, q_factor=1.333, record_duration=2.0)
        with pytest.warns(UserWarning, match="pinned"):
            fc, q, t = ram_band.resolve_broadband_grid(
                src, knobs=ram._knob_record(), log=ram._log)
        assert fc == pytest.approx(200.0)    # the middle array bin (odd count)
        assert (q, t) == (1.333, 2.0)

    def test_non_uniform_spacing_raises(self):
        freqs = np.array([50.0, 60.0, 80.0, 200.0, 350.0])
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        ram = self.RAM(verbose=False)
        with pytest.raises(ConfigurationError, match='non-uniform'):
            ram_band.resolve_broadband_grid(src, knobs=ram._knob_record(),
                                            log=ram._log)

    def test_non_uniform_spacing_passes_where_bins_march_one_by_one(self):
        """The Collins BROADBAND loop (``require_uniform=False``) takes an
        increasing non-uniform array and has no (Q, T) for it; an unsorted
        one still refuses."""
        ram = self.RAM(verbose=False)
        src = uacpy.Source(depths=25.0,
                           frequencies=[50.0, 60.0, 80.0, 200.0, 350.0])
        fc, Q, T = ram_band.resolve_broadband_grid(src, require_uniform=False,
                                                   knobs=ram._knob_record(),
                                                   log=ram._log)
        assert (fc, Q, T) == (80.0, None, None)
        assert list(
            ram_band.requested_broadband_bins(src,
                                              knobs=ram._knob_record())) == [
            50.0, 60.0, 80.0, 200.0, 350.0]
        unsorted = uacpy.Source(depths=25.0,
                                frequencies=[50.0, 80.0, 60.0, 200.0])
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            ram_band.resolve_broadband_grid(unsorted, require_uniform=False,
                                            knobs=ram._knob_record(),
                                            log=ram._log)

    @pytest.mark.parametrize('pin', [dict(q_factor=100.0), dict(record_duration=5.0)])
    def test_a_lone_pin_beside_an_array_is_ignored_and_named(self, pin):
        """With a frequency array, one pinned knob cannot describe the band
        (``Q=100`` alone would march 99-101 Hz for a 90-110 Hz request):
        it is ignored, the warning names it, and the resolved (Q, T) are the
        array's own, so every backend marches and stamps the same band."""
        freqs = np.linspace(90.0, 110.0, 21)
        src = uacpy.Source(depths=25.0, frequencies=freqs)
        model = self.RAM(verbose=False)
        free = ram_band.resolve_broadband_grid(src, knobs=model._knob_record(),
                                               log=model._log)
        knob = next(iter(pin))
        with pytest.warns(UserWarning, match=rf"{knob}=\S+ is pinned alone"):
            model = self.RAM(verbose=False, **pin)
            got = ram_band.resolve_broadband_grid(src,
                                                  knobs=model._knob_record(),
                                                  log=model._log)
        assert got == free
        marched = ram_band.broadband_frequencies(*got)
        assert all(np.isclose(marched, f).any() for f in freqs)

    def test_single_freq_partial_pin_fills_in_broadband_defaults(self):
        """ram.md §5: with exactly one of Q/T pinned, the other fills in
        from the default band every engine shares (Q = 4, its half-width;
        T = 127/(fc·0.5), its bin spacing) — not the narrowband 1e6 / 1.0
        collapse — and the warning names which value came from it."""
        with pytest.warns(UserWarning, match='the default band'):
            model = self.RAM(verbose=False, record_duration=5.0)
            fc, Q, T = ram_band.resolve_broadband_grid(
                self.source_scalar, knobs=model._knob_record(), log=model._log)
        assert (fc, Q, T) == (F_CENTER, 4.0, 5.0)
        with pytest.warns(UserWarning, match='the default band'):
            model = self.RAM(verbose=False, q_factor=8.0)
            fc, Q, T = ram_band.resolve_broadband_grid(
                self.source_scalar, knobs=model._knob_record(), log=model._log)
        assert (fc, Q) == (F_CENTER, 8.0)
        assert T == pytest.approx(127.0 / (F_CENTER * 0.5))


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
            speeds=SoundSpeeds(surface=1500.0),
            metadata={'grn_file': '/pinned/model.grn'},
        )

    def test_synthesize_time_series_keeps_source_metadata(self):
        tf = self._tf_with_metadata()
        wf = np.zeros(int(0.05 * FS))
        wf[: int(0.005 * FS)] = 1.0
        ts = tf.synthesize_time_series(source_waveform=wf, sample_rate=FS)
        assert ts.metadata['grn_file'] == '/pinned/model.grn'
        assert ts.speeds.surface == 1500.0
        assert ts.synthesis_window is None          # synthesis keys too

    def test_to_time_trace_keeps_source_metadata(self):
        tf = self._tf_with_metadata()
        trace = tf.to_time_trace(depth=25.0, range=100.0)
        assert trace.metadata['grn_file'] == '/pinned/model.grn'
        assert trace.speeds.surface == 1500.0
        assert trace.synthesis_window == 'hann'


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
        trace = tf.to_time_trace(t_start=0.0)
        assert trace.pinned['depth'] == 20.0
        assert trace.pinned['range'] == 500.0
        # The (20 m, 500 m) cell carries |H| = 11 against 2 at
        # (10 m, 1000 m); the same synthesis on both cells (one record
        # placement, t_start=0) preserves that amplitude ratio, so the
        # defaulted pick is visible in the data too.
        other = tf.to_time_trace(depth=10.0, range=1000.0, t_start=0.0)
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
        ``c_max``/``c0``, so ``_warn_unsolved_bins`` and the window-anchor
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
        with recorded_warnings() as record:
            self._tf_nan_cell_no_stamped_speed().to_time_trace(
                depth=20.0, range=2000.0)
        messages = [str(w.message) for w in record]
        assert any('entirely NaN' in m for m in messages), messages
        assert any('stamped no sound speed' in m for m in messages), messages
        assert all(m.startswith('to_time_trace: ') for m in messages), messages

    def test_synthesize_time_series_warnings_carry_its_label(self):
        wf = np.zeros(64)
        wf[:8] = 1.0
        with recorded_warnings() as record:
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
            speeds=SoundSpeeds(surface=C_WATER),
        )

    def test_short_window_warning_carries_the_entry_points_label(self):
        with recorded_warnings() as record:
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


class TestTheSynthesisWindowAnchorsOnTheStampedSpeed:
    """The window opens at the estimated first arrival r/c_fast. A model that
    stamps a c_max BELOW the 1500 m/s default (cold or fresh water) must
    anchor on it: anchored on 1500 the window opens early by
    r·(1/c_max − 1/1500), the arrival wraps a whole record and the time axis
    labels it one record early, with nothing to show for it."""

    R = 100_000.0

    def _tf(self, speeds=None):
        from uacpy.core.results import Field, PhaseReference
        freqs = np.arange(100.0, 150.0 + 1e-9, 0.5)          # Δf = 0.5 Hz: a 2 s record
        H = np.exp(-2j * np.pi * freqs * self.R / 1450.0)
        return Field(
            data=H[None, None, :],
            coords={'depth': np.array([50.0]), 'range': np.array([self.R]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([50.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            speeds=speeds,
        )

    def _peak_time(self, tf):
        with recorded_warnings() as rec:
            trace = tf.to_time_trace(window=None)
        t = np.asarray(trace.coords['time'])
        p = np.abs(np.asarray(trace.data))
        return t, float(t[np.argmax(p)]), [str(w.message) for w in rec]

    def test_a_stamped_speed_below_the_default_anchors_the_window(self):
        t, peak, msgs = self._peak_time(self._tf(SoundSpeeds(water_max=1450.0)))
        truth = self.R / 1450.0                                  # 68.966 s
        assert t[0] <= truth <= t[-1], (t[0], truth, t[-1])
        assert abs(peak - truth) < 0.005, (peak, truth)          # half a 125 Hz cycle
        assert not any('stamped no sound speed' in m for m in msgs)

    def test_with_nothing_stamped_the_default_anchors_and_warns(self):
        t, peak, msgs = self._peak_time(self._tf())
        truth = self.R / 1450.0
        # Anchored on 1500 the 2 s window opens at 65.67 s and closes before
        # 68.97 s: the wrap is the pre-existing, WARNED behaviour.
        assert any('stamped no sound speed' in m for m in msgs)
        assert not any('the model stamped' in m for m in msgs)
        assert any('BeamformedField carries none' in m for m in msgs)
        assert not (t[0] <= truth <= t[-1])

    def test_a_shared_window_uses_the_same_anchor(self):
        tf = self._tf(SoundSpeeds(water_max=1450.0))
        fs = 1000.0
        wf = np.zeros(int(0.05 * fs)); wf[0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = tf.synthesize_time_series(source_waveform=wf, sample_rate=fs)
        t = np.asarray(out.coords['time'])
        assert t[0] <= self.R / 1450.0 <= t[-1]


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
            speeds=SoundSpeeds(surface=c0))

    def test_warns_when_range_span_exceeds_window(self):
        tf = self._pure_delay_tf([100.0, 3000.0])   # 1.93 s spread, ~1 s window
        wf = np.zeros(64); wf[0] = 1.0
        with pytest.warns(UserWarning, match=r"range span|wrap"):
            tf.synthesize_time_series(wf, sample_rate=4000.0)

    def test_no_span_warning_for_single_range(self):
        tf = self._pure_delay_tf([100.0])
        wf = np.zeros(64); wf[0] = 1.0
        with recorded_warnings() as rec:
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


class TestSynthesisFloorsAtTheModelsTimeSampleCount:
    """The auto IFFT length is never below the time-sample count the model
    reports as :attr:`Field.synthesis_floor` (OASP's NX, OASSP's NT,
    mpiramS's Nsam). The 21 bins of 100-300 Hz at 10 Hz size an unfloored
    grid of 128 (4 bins per frequency, rounded up to a power of two)."""

    @staticmethod
    def _tf(floor=None, metadata=None):
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 300.0, 21)
        return Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': np.array([25.0]), 'range': np.array([100.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([25.0]),
            frequencies=freqs, phase_reference=PhaseReference.TRAVELLING_WAVE,
            synthesis_floor=floor, metadata=metadata)

    @pytest.mark.parametrize('floor,n_time', [
        (None, 128), (128, 128), (129, 256), (512, 512)])
    def test_the_trace_is_as_long_as_the_larger_of_the_two(self, floor,
                                                             n_time):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._tf(floor).to_time_trace(depth=25.0, range=100.0)
            series = self._tf(floor).synthesize_time_series(
                source_waveform=np.hanning(8), sample_rate=100.0)
        assert trace.coords['time'].size == n_time
        assert series.coords['time'].size == n_time

    def test_a_sample_count_in_metadata_does_not_floor_the_grid(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._tf(metadata={'n_samples': 512}).to_time_trace(
                depth=25.0, range=100.0)
        assert trace.coords['time'].size == 128


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
            speeds=SoundSpeeds(surface=C_WATER),
        )

    @pytest.mark.parametrize('nfft', [None, 2048, 4096, 8192])
    def test_flat_h_reproduces_unit_peak(self, nfft):
        fs = 2000.0
        t = np.arange(2000) / fs
        src = (np.exp(-0.5 * ((t - 0.5) / 0.05) ** 2)
               * np.cos(2 * np.pi * 100.0 * (t - 0.5)))
        ts = self._flat_tf().synthesize_time_series(
            src, fs, window=None, nfft=nfft, t_start=0.0,
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
        a = tf.to_time_trace(window=None, nfft=4096, t_start=0.0)
        b = tf.to_time_trace(window=None, nfft=8192, t_start=0.0)
        assert float(np.abs(a.data).max()) == pytest.approx(
            float(np.abs(b.data).max()), rel=1e-6,
        )


class TestSourceSpectrumAtArbitraryFrequencies:
    """``waveform_spectrum_at`` must be exact off the waveform's own DFT grid.

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
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        wf, fs = self._wf()
        grid = np.fft.rfftfreq(wf.size, 1.0 / fs)
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), np.fft.rfft(wf) / fs,
            rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize('shift', [0.5, 0.25])
    def test_exact_on_a_half_bin_offset_grid(self, shift):
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        wf, fs = self._wf()
        native = np.fft.rfftfreq(wf.size, 1.0 / fs)
        grid = native[:-1] + shift * (native[1] - native[0])
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), self._dtft(wf, fs, grid),
            rtol=1e-9, atol=1e-12)

    def test_exact_on_a_finer_grid(self):
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        wf, fs = self._wf()
        grid = np.fft.rfftfreq(4 * wf.size, 1.0 / fs)
        grid = grid[grid <= fs / 2]
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid), self._dtft(wf, fs, grid),
            rtol=1e-9, atol=1e-12)

    def test_out_of_band_frequencies_are_zero(self):
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        wf, fs = self._wf()
        out = _source_spectrum_at(wf, fs, np.array([-10.0, fs, 2 * fs]))
        assert np.all(out == 0)

    def test_chunking_does_not_change_the_result(self, monkeypatch):
        from uacpy.acoustic_signal import spectrum_at
        wf, fs = self._wf()
        grid = np.linspace(10.0, 900.0, 137)
        whole = spectrum_at.waveform_spectrum_at(wf, fs, grid)
        monkeypatch.setattr(spectrum_at, '_SCRATCH_BLOCK_ELEMS', 1000)
        np.testing.assert_allclose(
            spectrum_at.waveform_spectrum_at(wf, fs, grid), whole,
            rtol=1e-12, atol=1e-15)

    def test_a_complex_waveform_keeps_its_imaginary_part(self):
        """A complex baseband tone ``exp(i 2 pi 100 t)`` over 1 s has
        ``S(100) = 1`` and nothing at -100 Hz; casting it to real halves
        ``S(100)`` to 0.5 and puts the other half at -100 Hz."""
        from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
        fs = 1000.0
        t = np.arange(1000) / fs
        s = waveform_spectrum_at(np.exp(2j * np.pi * 100.0 * t), fs,
                                 [100.0, -100.0, -250.0])
        assert s[0] == pytest.approx(1.0, abs=1e-9)
        assert abs(s[1]) < 1e-9
        s_neg = waveform_spectrum_at(np.exp(-2j * np.pi * 100.0 * t), fs,
                                     [-100.0, 100.0, -fs])
        assert s_neg[0] == pytest.approx(1.0, abs=1e-9)
        assert abs(s_neg[1]) < 1e-9 and s_neg[2] == 0
        # The dense path (a non-uniform pair) agrees with the contour path.
        dense = waveform_spectrum_at(np.exp(-2j * np.pi * 100.0 * t), fs,
                                     [-100.0, 37.3, 400.0])
        assert dense[0] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize('bad', [
        dict(sample_rate=0.0), dict(sample_rate=np.nan),
        dict(waveform=np.array([])), dict(waveform=np.array([1.0, np.nan])),
        dict(freqs=[np.inf])])
    def test_bad_inputs_are_refused_by_name(self, bad):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
        kw = dict(waveform=np.ones(16), sample_rate=1000.0, freqs=[100.0])
        kw.update(bad)
        name = next(iter(bad))
        match = ('waveform_spectrum_at: waveform ' if name == 'waveform'
                 else 'waveform_spectrum_at')
        with pytest.raises(ConfigurationError, match=match):
            waveform_spectrum_at(kw['waveform'], kw['sample_rate'],
                                 kw['freqs'])


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
    scipy's ``czt`` sums exactly the DTFT ``waveform_spectrum_at`` documents,
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
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        wf = self._waveform(n_wf)
        ref = _outer_product_dtft(wf, fs, grid)
        got = _source_spectrum_at(wf, fs, grid)
        assert got.shape == ref.shape
        assert _rel_norm(ref, got) < 1e-9
        # The out-of-band zeros are exact zeros on both routes, not small.
        np.testing.assert_array_equal(got == 0, ref == 0)

    def test_an_all_zero_waveform_returns_exact_zeros(self):
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
        got = _source_spectrum_at(np.zeros(64), 1000.0,
                                  np.linspace(0.0, 400.0, 11))
        assert np.array_equal(got, np.zeros(11, dtype=np.complex128))

    def test_a_non_uniform_grid_keeps_the_dense_sum(self):
        from uacpy.acoustic_signal.spectrum_at import (
            _chirp_step, waveform_spectrum_at as _source_spectrum_at,
        )
        wf, fs = self._waveform(256), 2000.0
        grid = np.geomspace(20.0, 900.0, 64)          # ascending, not uniform
        np.testing.assert_allclose(
            _source_spectrum_at(wf, fs, grid),
            _outer_product_dtft(wf, fs, grid), rtol=1e-13, atol=1e-16)
        assert _chirp_step(grid, wf.size, fs) is None

    def test_the_dense_fallback_chunks_over_frequency(self, monkeypatch):
        # ``_SCRATCH_BLOCK_ELEMS`` bounds the phase matrix, and only the dense
        # path builds one — so the block arithmetic is exercised on a grid the
        # contour cannot serve rather than on the uniform one above, where
        # both calls would take the chirp-z route and agree vacuously.
        from uacpy.acoustic_signal import spectrum_at
        wf, fs = self._waveform(256), 2000.0
        grid = np.geomspace(20.0, 900.0, 137)
        whole = spectrum_at.waveform_spectrum_at(wf, fs, grid)
        monkeypatch.setattr(spectrum_at, '_SCRATCH_BLOCK_ELEMS', 1000)
        np.testing.assert_allclose(
            spectrum_at.waveform_spectrum_at(wf, fs, grid), whole,
            rtol=1e-13, atol=1e-16)

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
        from uacpy.acoustic_signal.spectrum_at import _chirp_step
        grid = np.linspace(100.0, 900.0, 401)
        grid[200] += 1e-4                       # 1e-7 of the span
        assert _chirp_step(grid, 4096, 8000.0) is None
        assert _chirp_step(np.linspace(100.0, 900.0, 401), 4096, 8000.0) \
            == pytest.approx(2.0)

    def test_the_synthesised_trace_matches_the_dense_sum(self, monkeypatch):
        # End to end through the public entry point, against the same Field
        # synthesised with the dense sum forced back in.
        import uacpy.acoustic_signal.spectrum_at as S
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
        monkeypatch.setattr(S, '_chirp_step', lambda *a, **k: None)
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
        # The deferred import resolves the attribute at CALL time, so
        # the patch sits on the module that now owns the function.
        import uacpy.acoustic_signal.spectrum_at as S
        monkeypatch.setattr(S, 'waveform_spectrum_at', _never_called)
        tf = self._tf([100.0, 110.0, 130.0, 140.0])
        with pytest.raises(ConfigurationError, match='uniformly spaced'):
            tf.synthesize_time_series(source_waveform=np.ones(64),
                                      sample_rate=FS)

    def test_a_descending_axis_is_refused_the_same_way(self, monkeypatch):
        # The deferred import resolves the attribute at CALL time, so
        # the patch sits on the module that now owns the function.
        import uacpy.acoustic_signal.spectrum_at as S
        monkeypatch.setattr(S, 'waveform_spectrum_at', _never_called)
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
        from uacpy.core.results._field_synthesis import _ifft_to_trace
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


def test_the_auto_derived_grid_says_how_long_a_record_it_bought():
    """The derived grid sets the record, and the record comes from the SOURCE
    pulse — not from the channel. A pulse shorter than the multipath spread
    therefore folds the tail onto the early trace, and the announcement has
    to say so: quoting only the band and the spacing leaves the reader to
    work out that 1/df is the whole record they are getting. The number to
    quote is 1/df of the grid actually returned, which subdivision can make
    longer than the pulse — here the 350-650 Hz band of the 0.02 s burst
    (7 bins at 50 Hz) is subdivided to 9 bins, 1/37.5 Hz = 0.02667 s."""
    from uacpy.models._band import time_series_band

    fs, dur = 20000.0, 0.020            # 20 ms pulse -> a 20 ms record
    t = np.arange(0.0, dur, 1.0 / fs)
    wf = np.hanning(t.size) * np.sin(2 * np.pi * 500.0 * t)
    text = time_series_band(wf, fs, model_name='Bellhop').notice
    assert text, "the derivation announced nothing"
    assert 'record' in text, text
    assert '0.02667 s' in text, f"the 0.02667 s record is not named: {text}"
    assert 'output_duration' in text, text


@pytest.mark.requires_binary  # runs a model
def test_auto_derived_timeseries_grid_resolves_the_band():
    """A 20 ms burst gives Delta f = 50 Hz, so its 400-600 Hz band derives
    only 5 bins — too few for the frequency-axis taper to leave an interior,
    which collapses the trace to a CW tone. The derived grid must carry
    enough bins to represent an arrival: the envelope over the record is not
    flat."""
    import warnings as _w
    from scipy.signal import hilbert
    from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties,
                       Source, Receiver, Kraken)
    from uacpy.core.run_settings import RunMode

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
    assert r.run_settings.frequencies.size >= 9
    assert envelope.min() / envelope.max() < 0.5, (
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
        from uacpy.core.results._field_synthesis import _ifft_to_trace
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
            window=None, nfft=None, t_start=0.0)
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
                                            window=None)
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
        from uacpy.models._band import time_series_band
        t, wf = self._source()
        derived = time_series_band(wf, self.FS,
                                   model_name='Bellhop').frequencies
        freqs = np.linspace(derived[0], derived[-1], 8)
        assert abs(self._bin_offset(freqs)) > 0.5, "grid is already aligned"

        tf = uacpy.Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': [10.0], 'range': [0.0], 'frequency': freqs},
            model='X', frequencies=freqs, phase_reference='travelling_wave')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = tf.synthesize_time_series(wf, self.FS, t_start=0.0,
                                            window=None)
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

    def _trace(self, speeds, range_m=60000.0, metadata=None):
        tf = uacpy.Field(
            data=np.ones((1, 1, self.FREQS.size), dtype=complex),
            coords={'depth': [100.0], 'range': [range_m],
                    'frequency': self.FREQS},
            model='RAM', frequencies=self.FREQS,
            phase_reference='travelling_wave', speeds=speeds,
            metadata=metadata)
        with recorded_warnings() as caught:
            trace = tf.to_time_trace()
        return float(trace.coords['time'][0]), caught

    def test_water_max_anchors_before_the_earliest_arrival(self):
        t0, caught = self._trace(SoundSpeeds(water_min=1500.0,
                                             water_max=1550.0,
                                             surface=1520.0))
        assert t0 <= 60000.0 / 1550.0
        assert not [w for w in caught if 'wrap to the end' in str(w.message)]

    def test_missing_water_max_warns_at_long_range(self):
        _, caught = self._trace(SoundSpeeds(water_min=1500.0, surface=1520.0))
        assert [w for w in caught if 'wrap to the end' in str(w.message)], (
            "a long-range trace with no c_max must say the window start is "
            "an estimate")

    def test_short_range_does_not_warn(self):
        t0, caught = self._trace(SoundSpeeds(surface=1500.0), range_m=2000.0)
        assert t0 <= 2000.0 / 1500.0
        assert not [w for w in caught if 'wrap to the end' in str(w.message)]

    def test_water_min_never_binds_the_anchor(self):
        # Only fastest-speed candidates may anchor the window: r/c_min is an
        # upper bound on the arrival, and anchoring on it opens the window
        # early enough that the true arrival wraps to the end of the record.
        t0, _ = self._trace(SoundSpeeds(water_min=5000.0, water_max=1550.0))
        assert t0 == pytest.approx(60000.0 / 1550.0 - 0.1, abs=1e-9)

    def test_pe_reference_speed_never_binds_the_anchor(self):
        # RAM stamps its Padé expansion point as 'pe_reference_speed'; it is
        # an algorithmic constant, often above every physical speed, so it
        # must not enter the physical-speed max.
        t0, _ = self._trace(SoundSpeeds(water_max=1550.0),
                            metadata={'pe_reference_speed': 1700.0})
        assert t0 == pytest.approx(60000.0 / 1550.0 - 0.1, abs=1e-9)


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
            speeds=SoundSpeeds(surface=1500.0, water_max=1520.0))

    def test_to_time_trace_warns_and_returns_nan(self):
        tf = self._tf(np.full((1, 1, 16), np.nan, dtype=complex))
        with pytest.warns(UserWarning, match='entirely NaN'):
            trace = tf.to_time_trace()
        assert np.all(np.isnan(trace.data))

    def test_isolated_unsolved_bins_make_the_trace_no_data(self):
        """Reversed policy: a NaN bin used to be zero-filled and the trace
        came back finite, which put a notch the model never produced into a
        waveform that looked ordinary. An unsolved bin is no data, and a
        trace cannot be synthesised from a spectrum with a hole in it."""
        rng = np.random.default_rng(0)
        H = (rng.standard_normal((1, 1, 16))
             + 1j * rng.standard_normal((1, 1, 16)))
        H[0, 0, 3] = np.nan
        with pytest.warns(UserWarning, match='did not solve'):
            trace = self._tf(H).to_time_trace()
        assert np.all(np.isnan(trace.data))

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
        from uacpy.core.results._field_synthesis import _ifft_to_trace
        from uacpy.acoustic_signal.spectrum_at import (
            waveform_spectrum_at as _source_spectrum_at,
        )
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
            speeds=SoundSpeeds(surface=1500.0, water_max=1520.0))
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
                    source_spectrum=src, window=None, nfft=nfft,
                    t_start=t_start)
                np.testing.assert_allclose(
                    out.data[di, ri], tr.data, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(out.data)) > 0.0


def _waveform():
    return np.sin(2.0 * np.pi * 100.0 * np.arange(64) / 1000.0)


@pytest.mark.requires_binary
class TestAComplexPulseIsRefusedByTheOneWaveformRule:
    """A model's TIME_SERIES run applies the one waveform rule
    (``source_waveform_problem``) that every waveform entry point does: a
    real, finite, 1-D array. A complex waveform is refused even when its
    imaginary part is zero, and the time settings then store no pulse; a
    real waveform is stored as its float64 samples."""

    @staticmethod
    def _time(wf):
        env, source, receiver = _make_env()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            return Bellhop(verbose=False).run_settings(
                env, source, receiver, RunMode.TIME_SERIES,
                source_waveform=wf, sample_rate=1000.0).time

    @pytest.mark.parametrize('kind', ['list', 'ndarray'])
    def test_a_complex_waveform_with_zero_imaginary_part_is_refused(
            self, kind):
        wf = [complex(v, 0.0) for v in _waveform()]
        if kind == 'ndarray':
            wf = np.asarray(wf)
        with pytest.raises(ConfigurationError, match=r'np\.real'):
            Bellhop(verbose=False)._require_timeseries_signal(
                RunMode.TIME_SERIES, wf, 1000.0)

    def test_a_real_waveform_runs_unchanged(self):
        pulse = self._time(_waveform()).source_waveform
        assert pulse.dtype == np.float64
        assert np.array_equal(pulse, _waveform())

    @pytest.mark.parametrize('bad', ['tuple', 'two_rows', 'column'])
    def test_a_waveform_that_is_not_one_dimensional_is_refused(self, bad):
        wf = np.asarray(_waveform())
        wf = {'tuple': (np.arange(wf.size) / 1000.0, wf),
              'two_rows': np.vstack([wf, wf]),
              'column': wf[:, None]}[bad]
        with pytest.raises(ConfigurationError, match=r'lfm_chirp\(\.\.\.\)\[1\]'):
            Bellhop(verbose=False)._require_timeseries_signal(
                RunMode.TIME_SERIES, wf, 1000.0)

    @pytest.mark.parametrize('mode,wf,rate', [
        (RunMode.TIME_SERIES, _waveform(), 1000.0),
        (RunMode.BROADBAND, None, None)])
    def test_the_guard_refuses_and_returns_nothing(self, mode, wf, rate):
        assert Bellhop(verbose=False)._require_timeseries_signal(
            mode, wf, rate) is None


class TestUnsolvedBinsDoNotBecomeSilence:
    """A NaN bin in H(f) is a frequency the model did not solve, not a
    frequency carrying no energy. Zero-filling it synthesises a spectral
    notch into an otherwise finite-looking waveform, so the trace built from
    an incomplete spectrum is no data — and the gap is named."""

    @staticmethod
    def _tf_with_one_unsolved_bin():
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)
        data = np.ones((1, 1, freqs.size), dtype=complex)
        data[0, 0, 4] = np.nan          # one frequency failed to solve
        return Field(
            data=data,
            coords={'depth': np.array([20.0]),
                    'range': np.array([1000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )

    def test_a_single_unsolved_bin_makes_the_trace_no_data(self):
        with pytest.warns(UserWarning, match='did not solve'):
            trace = self._tf_with_one_unsolved_bin().to_time_trace(
                depth=20.0, range=1000.0)
        assert np.isnan(np.asarray(trace.data)).all()

    def test_the_warning_counts_the_unsolved_bins(self):
        with recorded_warnings() as record:
            self._tf_with_one_unsolved_bin().to_time_trace(
                depth=20.0, range=1000.0)
        msgs = [str(w.message) for w in record if 'did not solve' in str(w.message)]
        assert msgs and '1 of 9' in msgs[0], msgs

    def test_a_fully_solved_spectrum_synthesises_a_finite_trace(self):
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)
        f = Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': np.array([20.0]),
                    'range': np.array([1000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = f.to_time_trace(depth=20.0, range=1000.0)
        assert np.isfinite(np.asarray(trace.data)).all()




# ── The synthesis window defaults to no window whenever a waveform is given ──
# A window across the band of H(f) is a filter the channel does not contain:
# with a flat H = 1 it must leave the transmitted pulse's energy unchanged,
# and a Hann there removes 0.8-17 dB depending on where the pulse sits in the
# band. The bare impulse response keeps the Hann, which suppresses band-edge
# ringing.

from uacpy.core.results import Field, PhaseReference  # noqa: E402
from uacpy.models.ram import _band as ram_band

_WFS = 16000.0
def _wd_flat_field(f_lo, f_hi, n):
    freqs = np.linspace(f_lo, f_hi, n)
    return Field(
        data=np.ones((1, 1, n), dtype=complex),
        coords={'depth': np.array([10.0]), 'range': np.array([0.0]),
                'frequency': freqs},
        model='Synthetic', source_depths=np.array([10.0]), frequencies=freqs,
        phase_reference=PhaseReference.TRAVELLING_WAVE,
        speeds=SoundSpeeds(surface=1500.0),
    )


def _wd_burst(f0=500.0, dur=0.01, total=0.1):
    t = np.arange(int(total * _WFS)) / _WFS
    x = np.zeros_like(t)
    n = int(dur * _WFS)
    x[:n] = np.hanning(n) * np.sin(2 * np.pi * f0 * t[:n])
    return x


def _wd_energy_dB(x, dt):
    return 10 * np.log10(np.sum(np.asarray(x) ** 2) * dt)


class TestWaveformSynthesisIsUnwindowedByDefault:
    def test_flat_channel_returns_the_pulse_energy(self):
        wf = _wd_burst()
        tf = _wd_flat_field(10.0, 4000.0, 400)
        ts = tf.synthesize_time_series(wf, _WFS, t_start=0.0)
        t = ts.coords['time']
        got = _wd_energy_dB(ts.data[0, 0], t[1] - t[0])
        assert got == pytest.approx(_wd_energy_dB(wf, 1 / _WFS), abs=0.05)
        assert ts.synthesis_window is None

    def test_a_hann_window_would_bias_the_same_pulse_low(self):
        wf = _wd_burst()
        tf = _wd_flat_field(10.0, 4000.0, 400)
        ts = tf.synthesize_time_series(wf, _WFS, t_start=0.0, window='hann')
        t = ts.coords['time']
        loss = _wd_energy_dB(wf, 1 / _WFS) - _wd_energy_dB(ts.data[0, 0], t[1] - t[0])
        assert loss > 5.0

    def test_to_time_trace_with_a_waveform_is_unwindowed(self):
        wf = _wd_burst()
        tf = _wd_flat_field(10.0, 4000.0, 400)
        tr = tf.to_time_trace(source_waveform=wf, sample_rate=_WFS, t_start=0.0)
        assert tr.synthesis_window is None
        t = tr.coords['time']
        assert _wd_energy_dB(tr.data, t[1] - t[0]) == pytest.approx(
            _wd_energy_dB(wf, 1 / _WFS), abs=0.05)

    def test_bare_impulse_response_keeps_the_hann(self):
        tf = _wd_flat_field(100.0, 1000.0, 91)
        tr = tf.to_time_trace(t_start=0.0)
        assert tr.synthesis_window == 'hann'

    def test_an_explicit_none_is_rectangular_with_or_without_a_source(self):
        """``window='auto'`` decides from the source; ``None`` never does:
        it is the rectangular window on the bare impulse response too."""
        tf = _wd_flat_field(100.0, 1000.0, 91)
        assert tf.to_time_trace(t_start=0.0, window=None).synthesis_window is None
        wf = _wd_burst()
        tf = _wd_flat_field(10.0, 4000.0, 400)
        tr = tf.to_time_trace(source_waveform=wf, sample_rate=_WFS, t_start=0.0,
                              window=None)
        assert tr.synthesis_window is None


class TestBandEdgeRingingWarning:
    def test_band_inside_the_pulse_spectrum_warns(self):
        wf = _wd_burst()
        tf = _wd_flat_field(450.0, 550.0, 11)   # cuts the burst's main lobe
        with pytest.warns(UserWarning, match="cuts the source spectrum"):
            tf.synthesize_time_series(wf, _WFS, t_start=0.0)

    def test_band_holding_the_pulse_spectrum_is_silent(self):
        wf = _wd_burst()
        tf = _wd_flat_field(10.0, 4000.0, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            tf.synthesize_time_series(wf, _WFS, t_start=0.0)

    def test_an_explicit_window_does_not_warn_about_ringing(self):
        wf = _wd_burst()
        tf = _wd_flat_field(450.0, 550.0, 11)
        with recorded_warnings() as rec:
            tf.synthesize_time_series(wf, _WFS, t_start=0.0, window='hann')
        assert not any("cuts the source spectrum" in str(w.message)
                       for w in rec)

    @staticmethod
    def _edge_warnings(edge):
        """Warnings from the band-edge check on a spectrum whose two edges
        sit at ``edge`` re its unit peak, every RuntimeWarning an error."""
        from uacpy.acoustic_signal._synthesis import (
            warn_band_edge_cuts_spectrum)
        spectrum = np.concatenate([[edge], np.ones(9), [edge]])
        with recorded_warnings() as rec:
            warnings.simplefilter("error", RuntimeWarning)
            warn_band_edge_cuts_spectrum(spectrum, None, 'probe')
        return [str(w.message) for w in rec]

    def test_a_spectrum_zero_at_its_edges_is_not_cut(self):
        """Nothing is cut where the spectrum is already zero, and the zero
        edge takes no logarithm (a divide-by-zero RuntimeWarning escaped)."""
        assert self._edge_warnings(0.0) == []

    def test_a_small_nonzero_edge_below_the_threshold_is_silent(self):
        assert self._edge_warnings(1e-3) == []          # -60 dB

    def test_an_edge_at_the_threshold_is_silent(self):
        assert self._edge_warnings(0.01) == []          # -40.0 dB exactly

    def test_an_edge_above_the_threshold_warns_with_its_level(self):
        messages = self._edge_warnings(0.0101)           # -39.9 dB
        assert len(messages) == 1
        assert "cuts the source spectrum at -39.9 dB" in messages[0]

    def test_a_hann_spectrum_through_a_trace_raises_no_runtime_warning(self):
        tf = _wd_flat_field(450.0, 550.0, 11)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            tf.to_time_trace(source_spectrum=np.hanning(11), t_start=0.0)


class TestEveryWaveformEntryPointAppliesTheOneRule:
    """Decision 38: the same waveform is accepted or refused alike by every
    entry point that takes one — the five Field methods and the plain-array
    ``synthesize_time_series``."""

    _FS = 4000.0

    @classmethod
    def _entry_points(cls):
        from uacpy.acoustic_signal import synthesize_time_series
        tf = _wd_flat_field(100.0, 900.0, 81)
        fs = cls._FS
        return {
            'to_time_trace': lambda w: tf.to_time_trace(
                source_waveform=w, sample_rate=fs, t_start=0.0),
            'synthesize_time_series': lambda w: tf.synthesize_time_series(
                w, fs, t_start=0.0),
            'broadband_loss': lambda w: tf.broadband_loss(
                source_waveform=w, sample_rate=fs),
            'sound_exposure_level': lambda w: tf.sound_exposure_level(
                w, fs, t_start=0.0),
            'peak_sound_pressure_level':
                lambda w: tf.peak_sound_pressure_level(w, fs, t_start=0.0),
            'array synthesize_time_series': lambda w: synthesize_time_series(
                tf.data, frequencies=tf.coords['frequency'], source_waveform=w, sample_rate=fs),
        }

    @staticmethod
    def _pulse():
        t = np.arange(80) / 4000.0
        return np.sin(2 * np.pi * 500.0 * t) * np.hanning(t.size)

    @pytest.mark.parametrize('entry', [
        'to_time_trace', 'synthesize_time_series', 'broadband_loss',
        'sound_exposure_level', 'peak_sound_pressure_level',
        'array synthesize_time_series'])
    def test_a_real_one_dimensional_waveform_is_accepted(self, entry):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self._entry_points()[entry](self._pulse())

    @pytest.mark.parametrize('bad,match', [
        ('tuple', r'lfm_chirp\(\.\.\.\)\[1\]'),
        ('two_rows', r'lfm_chirp\(\.\.\.\)\[1\]'),
        ('one_row', r'lfm_chirp\(\.\.\.\)\[1\]'),
        ('column', r'lfm_chirp\(\.\.\.\)\[1\]'),
        ('complex', r'np\.real'),
        ('nan', 'non-finite')])
    @pytest.mark.parametrize('entry', [
        'to_time_trace', 'synthesize_time_series', 'broadband_loss',
        'sound_exposure_level', 'peak_sound_pressure_level',
        'array synthesize_time_series'])
    def test_every_other_waveform_is_refused_alike(self, entry, bad, match):
        wf = self._pulse()
        wf = {'tuple': (np.arange(wf.size) / self._FS, wf),
              'two_rows': np.vstack([wf, wf]),
              'one_row': wf[None, :],
              'column': wf[:, None],
              'complex': wf + 0j,
              'nan': np.where(np.arange(wf.size) == 3, np.nan, wf)}[bad]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(ConfigurationError, match=match):
                self._entry_points()[entry](wf)


class TestSubCutoffBinsAreNamedAsSuch:
    """A normal-mode model records ``Field.sub_cutoff_bins`` for bins
    below the lowest mode's cutoff and leaves them NaN in H(f). Synthesising
    that field by hand is all-NaN, and "re-run them" cannot help below
    cutoff, so the warning names the cutoff and the remedies that do: a band
    above it, Scooter, or RunMode.TIME_SERIES. Without the metadata the
    ordinary unsolved-bin wording stays."""

    @staticmethod
    def _tf(sub_cutoff_bins):
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)
        data = np.ones((1, 1, freqs.size), dtype=complex)
        data[0, 0, :2] = np.nan
        return Field(
            data=data,
            coords={'depth': np.array([20.0]), 'range': np.array([1000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs, sub_cutoff_bins=sub_cutoff_bins,
            phase_reference=PhaseReference.TRAVELLING_WAVE)

    @staticmethod
    def _messages(call):
        with recorded_warnings() as record:
            call()
        return ' '.join(str(w.message) for w in record)

    @pytest.mark.parametrize('route', ['trace', 'series'])
    def test_recorded_sub_cutoff_bins_name_the_cutoff_and_the_remedies(
            self, route):
        tf = self._tf(2)
        call = ((lambda: tf.to_time_trace(depth=20.0, range=1000.0))
                if route == 'trace' else
                (lambda: tf.synthesize_time_series(np.hanning(32), 2000.0)))
        text = self._messages(call)
        assert 'cutoff' in text and 'Scooter' in text
        assert 'RunMode.TIME_SERIES' in text
        assert 'Re-run them, or' not in text

    @pytest.mark.parametrize('count', [None, 0])
    def test_without_the_record_the_unsolved_wording_stays(self, count):
        tf = self._tf(count)
        text = self._messages(
            lambda: tf.to_time_trace(depth=20.0, range=1000.0))
        assert 'did not solve' in text and 'cutoff' not in text


class TestTransferFunctionRoundTripGrid:
    """``to_transfer_function`` returns the rfft bins ``k·Δf`` of the trace:
    the model's own axis when the band started on a multiple of ``Δf``, the
    integer-``Δf`` grid (with leakage) when it did not — the two measured
    cases the method's docstring quotes."""

    @staticmethod
    def _round_trip(f0):
        f = np.arange(f0, 1000.0, 1.0)
        H = (0.01 * np.exp(-2j * np.pi * f * 0.5)
             + 0.005 * np.exp(-2j * np.pi * f * 0.52))
        field = Field(data=H[None, None, :],
                      coords={'depth': np.array([10.0]),
                              'range': np.array([750.0]), 'frequency': f},
                      speeds=SoundSpeeds(water_max=1500.0), frequencies=f)
        back = field.to_time_trace(depth=10.0, range=750.0, window=None,
                                   t_start=0.0).to_transfer_function()
        fb = back.coords['frequency']
        truth = (0.01 * np.exp(-2j * np.pi * fb * 0.5)
                 + 0.005 * np.exp(-2j * np.pi * fb * 0.52))
        inner = (fb > 50.0) & (fb < 950.0)
        return fb, np.max(np.abs(back.data.ravel() - truth)[inner]) / 0.015

    def test_a_band_on_the_grid_round_trips_to_rounding(self):
        fb, err = self._round_trip(25.0)
        assert fb[0] == 25.0 and err < 1e-12

    def test_an_offset_band_comes_back_on_the_integer_grid_with_leakage(self):
        fb, err = self._round_trip(25.3)
        assert fb[0] == 26.0
        assert 1e-3 < err < 5e-3


@pytest.mark.requires_binary
def test_output_duration_pads_the_pulse_except_on_the_time_marcher():
    """``output_duration=`` zero-pads the pulse the time settings record,
    as the IFFT synthesis reads it (Scooter); SPARC's ``output_duration``
    is the end of its own record, so its settings keep the pulse its STSFIL
    carries, unpadded. Both record the duration asked for."""
    import uacpy
    from uacpy.models import SPARC, Scooter
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                            bottom=uacpy.BoundaryProperties(
                                acoustic_type='rigid'))
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
    fs = 4000.0
    t = np.arange(400) / fs
    pulse = np.sin(2 * np.pi * 100.0 * t) * np.hanning(t.size)
    kw = dict(run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
              sample_rate=fs, output_duration=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        padded = Scooter(verbose=False).run_settings(env, src, rcv, **kw).time
        marched = SPARC(verbose=False).run_settings(env, src, rcv, **kw).time
    assert padded.source_waveform.size == 2000
    assert marched.source_waveform.size == 400
    np.testing.assert_array_equal(marched.source_waveform, pulse)
    assert padded.output_duration == marched.output_duration == 0.5
