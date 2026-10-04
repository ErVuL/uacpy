"""``uacpy.acoustic_signal``: the generators, and the guards every door keeps.

Two jobs in one file, because the second one only means anything swept across
the whole surface:

* **The generators produce what they document** — chirps, tone bursts, Ricker
  and Gaussian pulses, the SPARC library, m-sequences and BPSK, and the noise
  synthesised to a target spectrum — rather than merely returning finite
  samples of the right length.
* **Every entry point refuses what it cannot represent.** A sweep bound below
  zero, a rate that is not positive or not finite, an empty axis, a generator
  asked for a frequency above Nyquist, an estimator handed a ``Field``: the
  guards are checked across all five sub-modules at once, so a new door that
  forgets one is visible here rather than at a caller.

The statistics themselves are in ``test_spectral_estimators.py``, the
frequency-response fit in ``test_frequency_response.py``.
"""

import warnings

import numpy as np
import pytest

from uacpy.acoustic_signal.frf import FRF
from uacpy.acoustic_signal.generate import (
    gaussian_pulse, hfm_chirp, lfm_chirp, ricker_wavelet, tone_burst,
)
from uacpy.acoustic_signal.generate import (
    make_bandlimited_noise, synthesize_noise_from_psd,
)
from uacpy.comms.constellations import (
    fsk_demodulate as _fsk_demodulate, fsk_modulate as _fsk_modulate,
)
from uacpy.core.exceptions import ConfigurationError

NAN = float('nan')
#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]
from uacpy.acoustic_signal.detect import (ambiguity_function,
                                          pulse_compression)
from uacpy.acoustic_signal.spectral import probabilistic_welch, welch
from uacpy.acoustic_signal.generate import bpsk_modulate
from uacpy.acoustic_signal.timefreq import (
    cwt, instantaneous_frequency, spectrogram,
)
from uacpy.acoustic_signal.gathers import (
    fk_transform, radon_transform as _radon_transform,
    taup_transform as _taup_transform,
)
from uacpy.acoustic_signal.timefreq import (
    cepstrum as _cepstrum, envelope as _envelope,
    wigner_ville as _wigner_ville,
)
from uacpy.acoustic_signal.spectral import constant_q as _constant_q_spectrum
from uacpy.acoustic_signal.dispersion import warp_signal as _warp_signal


def _band_exposure(data, sample_rate, **options):
    """The ISO-band sound exposure: the estimator's ``'exposure'`` scaling
    reported on a band ladder. Returns a ``SpectralEstimate`` whose ``power``
    is Pa²·s per band and whose ``bands`` are the ``(low, centre, high)``
    edges those values sit on.
    """
    from uacpy.acoustic_signal.spectral import sound_exposure
    return sound_exposure(data, sample_rate, **options)


class TestGenerators:
    """Pulse / chirp generators produce the documented waveform, not just
    finite samples."""

    @staticmethod
    def _instantaneous_frequency(s, dt):
        """f_inst(t) from the analytic signal's unwrapped phase, on the
        midpoint time grid of the finite difference."""
        from scipy.signal import hilbert
        phase = np.unwrap(np.angle(hilbert(s)))
        return np.diff(phase) / (2.0 * np.pi * dt)

    def test_gaussian_pulse_centred_at_delay(self):
        # dt = 1e-4 s puts delay (index 500) and delay+duration (index 600)
        # exactly on the grid, so the peak and the 1/e point are exact.
        time = np.linspace(0, 0.1, 1001)
        s = gaussian_pulse(time, delay=0.05, duration=0.01)
        assert len(s) == len(time)
        assert int(np.argmax(s)) == 500          # envelope peak at delay
        assert s[500] == 1.0
        # exp(-((t-delay)/duration)^2): one duration off the peak is 1/e.
        assert s[600] == pytest.approx(np.exp(-1.0), rel=1e-12)
        assert s[400] == pytest.approx(np.exp(-1.0), rel=1e-12)

    def test_lfm_chirp_sweeps_fmin_to_fmax(self):
        fs, freq_start, freq_end, T = 20_000.0, 500.0, 1500.0, 0.2
        t, s = lfm_chirp(freq_start=freq_start, freq_end=freq_end, duration=T, sample_rate=fs)
        assert len(t) == len(s)
        f_inst = self._instantaneous_frequency(s, 1.0 / fs)
        tm = 0.5 * (t[:-1] + t[1:])
        # Hilbert edge ripple contaminates the ends, so fit the (linear)
        # interior and read the endpoints off the fit: measured errors are
        # < 1e-4 Hz at both ends, so 1 Hz on a 1000 Hz sweep is generous.
        interior = (tm > 0.1 * T) & (tm < 0.9 * T)
        slope, intercept = np.polyfit(tm[interior], f_inst[interior], 1)
        assert intercept == pytest.approx(freq_start, abs=1.0)
        assert slope * T + intercept == pytest.approx(freq_end, abs=1.0)

    @pytest.mark.parametrize("chirp", [lfm_chirp, hfm_chirp])
    @pytest.mark.parametrize("T,fs", [(0.29, 100.0), (0.007, 44100.0),
                                      (0.1, 8000.0)])
    def test_chirp_time_axis_is_sampled_at_the_requested_rate(self, chirp,
                                                              T, fs):
        # 0.29 * 100 evaluates to 28.999999999999996: the sample count is
        # round(T * fs) and the spacing is 1/fs (tone_burst's rule), so the
        # waveform played back at sample_rate lasts T and sweeps at the
        # requested rate.
        t, s = chirp(5.0, 20.0, T, sample_rate=fs)
        assert t.size == s.size == round(T * fs)
        assert t[0] == 0.0
        assert np.allclose(np.diff(t), 1.0 / fs, rtol=1e-12, atol=0.0)

    def test_hfm_chirp_frequency_is_hyperbolic_in_time(self):
        # HFM == linear *period* modulation: 1/f_inst(t) is linear in t,
        # running from 1/freq_min to 1/freq_max (Abraham §8.3.6's pulse).
        fs, freq_start, freq_end, T = 20_000.0, 500.0, 1500.0, 0.2
        t, s = hfm_chirp(freq_start=freq_start, freq_end=freq_end, duration=T, sample_rate=fs)
        period = 1.0 / self._instantaneous_frequency(s, 1.0 / fs)
        tm = 0.5 * (t[:-1] + t[1:])
        interior = (tm > 0.1 * T) & (tm < 0.9 * T)
        slope, intercept = np.polyfit(tm[interior], period[interior], 1)
        # Endpoint errors measured at 7e-8 s (P(0)) and 3e-8 s (P(T));
        # 1e-5 s against a 1.3e-3 s period span is generous.
        assert intercept == pytest.approx(1.0 / freq_start, abs=1e-5)
        assert slope * T + intercept == pytest.approx(1.0 / freq_end, abs=1e-5)
        # Discriminating half: the period really is linear (residual < 2 %
        # of the span; an LFM period fitted the same way leaves ~20 %).
        resid = period[interior] - (slope * tm[interior] + intercept)
        assert np.max(np.abs(resid)) < 0.02 * (1.0 / freq_start - 1.0 / freq_end)

    def test_ricker_wavelet_peaks_at_requested_frequency(self):
        time = np.linspace(0, 0.1, 1024)
        f0 = 200.0
        s = ricker_wavelet(time, frequency=f0)
        assert len(s) == len(time)
        # Spectral peak at the nominal frequency (within one rFFT bin).
        freqs = np.fft.rfftfreq(len(time), time[1] - time[0])
        peak = freqs[int(np.argmax(np.abs(np.fft.rfft(s))))]
        assert abs(peak - f0) <= freqs[1]
        # Zero mean: the Ricker is the second derivative of a Gaussian and
        # the u = 2πFt − 8 centring makes the truncation at t=0 negligible
        # (measured mean -2.5e-9 against a 0.443 lobe).
        assert abs(s.mean()) < 1e-6
        # The central lobe at u = 0 (t = 4/(πF)) is a TROUGH of
        # 0.5·(−0.5)·√π = −0.25·√π ≈ −0.4431 — the docs' "−0.44" value.
        # abs=1e-3 covers the grid not sampling u = 0 exactly.
        i0 = int(np.argmin(s))
        assert s[i0] == pytest.approx(-0.25 * np.sqrt(np.pi), abs=1e-3)
        assert time[i0] == pytest.approx(4.0 / (np.pi * f0),
                                         abs=time[1] - time[0])

    def test_tone_burst_peaks_at_requested_frequency(self):
        f = 1000.0
        fs = 48_000.0
        t, s = tone_burst(frequency=f, n_cycles=20, sample_rate=fs)
        # FFT peak should sit at f within the resolution.
        S = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s), 1.0 / fs)
        peak = freqs[np.argmax(np.abs(S))]
        assert abs(peak - f) < (fs / len(s)) * 2

    def test_tone_burst_dt_equals_inverse_sample_rate(self):
        """``tone_burst`` builds ``time`` so ``dt == 1 / sample_rate``
        exactly, which keeps round-trip Fourier identities
        (``np.fft.rfftfreq(N, dt)``) honest."""
        fs = 48_000.0
        t, s = tone_burst(frequency=1000.0, n_cycles=5, sample_rate=fs)
        # Identical length.
        assert len(s) == len(t)
        # First sample sits at t=0 (no spurious offset).
        assert t[0] == 0.0
        # ``dt`` exact to float precision — no rescaling.
        dt = t[1] - t[0]
        assert dt == 1.0 / fs
        # Uniform spacing across the whole vector (tolerant of the
        # 1-ulp roundoff that ``np.diff`` introduces on a stride-built
        # array).
        np.testing.assert_allclose(np.diff(t), 1.0 / fs, rtol=1e-12, atol=0)


class TestProcessing:
    """Processing helpers don't blow up on synthetic signals."""

    def test_noise_at_a_level_is_band_limited(self):
        """The noise ``psd_level_dB`` asks for sits inside the requested band.

        Band occupancy and the sample-to-sample correlation both describe
        the noise directly; variance alone cannot tell noise from any other
        waveform.
        """
        fs, fc, bw = 48_000.0, 10_000.0, 10_000.0
        _, y = make_bandlimited_noise(fc, bw, 8192 / fs, sample_rate=fs,
                                      psd_level_dB=40.0)

        freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
        power = np.abs(np.fft.rfft(y)) ** 2
        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        assert power[in_band].sum() / power.sum() > 0.9
        assert freqs[np.argmax(power)] > fc - bw / 2

        # Any monotone or otherwise smooth waveform correlates ~1 at lag 1.
        assert abs(np.corrcoef(y[:-1], y[1:])[0, 1]) < 0.7

    def test_noise_realisations_are_seeded_and_per_channel(self):
        """``rng`` selects the realisation, and channels are independent.

        Array-gain checks need zero cross-channel correlation, so a shared
        realisation across columns is a defect.
        """
        kw = dict(fc=10_000.0, bandwidth=10_000.0, duration=4096 / 48_000.0,
                  sample_rate=48_000.0, psd_level_dB=40.0)
        first = make_bandlimited_noise(**kw, rng=np.random.default_rng(3))[1]
        again = make_bandlimited_noise(**kw, rng=np.random.default_rng(3))[1]
        other = make_bandlimited_noise(**kw, rng=np.random.default_rng(4))[1]
        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)

        block = make_bandlimited_noise(**kw, n_channels=4)[1]
        assert block.shape == (4096, 4)
        corr = np.corrcoef(block.T)
        assert np.max(np.abs(corr[~np.eye(4, dtype=bool)])) < 0.2

    @pytest.mark.parametrize('bad', [0, -1, 1.5])
    def test_n_channels_must_be_a_positive_integer(self, bad):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='n_channels'):
            make_bandlimited_noise(1000.0, 500.0, 0.1, sample_rate=10_000.0,
                                   n_channels=bad)

    def test_one_channel_is_one_dimensional_and_two_is_a_column_pair(self):
        kw = dict(fc=1000.0, bandwidth=500.0, duration=0.1,
                  sample_rate=10_000.0)
        assert make_bandlimited_noise(**kw, n_channels=1)[1].shape == (1000,)
        assert make_bandlimited_noise(**kw, n_channels=2)[1].shape == (1000, 2)

    def test_make_bandlimited_noise_runs(self):
        # Returns (time, signal) like the other generators.
        fs, dur = 10_000.0, 0.1
        t, n = make_bandlimited_noise(
            fc=1000.0, bandwidth=500.0,
            duration=dur, sample_rate=fs,
        )
        assert len(n) > 0
        assert n.shape == t.shape == (int(dur * fs),)
        assert np.all(np.isfinite(n))
        assert t[0] == 0.0 and np.allclose(np.diff(t), 1.0 / fs)

    def test_without_a_level_the_noise_is_unit_rms(self):
        """``psd_level_dB=None`` is the unit-RMS contract, per channel."""
        kw = dict(fc=1000.0, bandwidth=200.0, duration=1.0,
                  sample_rate=8000.0)
        assert np.std(make_bandlimited_noise(
            **kw, rng=np.random.default_rng(0))[1]) == pytest.approx(1.0)
        assert np.std(make_bandlimited_noise(
            **kw, n_channels=3, rng=np.random.default_rng(0))[1],
            axis=0) == pytest.approx(np.ones(3))

    def test_noise_generators_are_reproducible_from_a_seeded_rng(self):
        """Every noise generator takes an ``rng=``, like the ``uacpy.comms``
        side, so a realisation can be reproduced independently of global
        numpy state."""
        from uacpy.acoustic_signal.generate import (
            synthesize_noise_from_psd)
        fs, dur = 10_000.0, 0.1
        kw = dict(fc=1000.0, bandwidth=500.0, duration=dur, sample_rate=fs)
        a = make_bandlimited_noise(**kw, rng=np.random.default_rng(7))[1]
        b = make_bandlimited_noise(**kw, rng=np.random.default_rng(7))[1]
        c = make_bandlimited_noise(**kw, rng=np.random.default_rng(8))[1]
        assert np.array_equal(a, b) and not np.array_equal(a, c)

        lkw = dict(kw, psd_level_dB=80.0, n_channels=2)
        assert np.array_equal(
            make_bandlimited_noise(**lkw, rng=np.random.default_rng(7))[1],
            make_bandlimited_noise(**lkw, rng=np.random.default_rng(7))[1])

        f = np.logspace(1, 3, 32)
        pkw = dict(duration=0.05, nfft=1024, sample_rate=fs)
        assert np.array_equal(
            synthesize_noise_from_psd(1e-6 / (1 + (f / 100) ** 2), f, **pkw,
                                      rng=np.random.default_rng(7))[1],
            synthesize_noise_from_psd(1e-6 / (1 + (f / 100) ** 2), f, **pkw,
                                      rng=np.random.default_rng(7))[1])


class TestDecidecadeBands:
    def test_standard_iso_centre_frequencies_and_ratio(self):
        from uacpy.acoustic_signal.bands import decidecade_bands
        lo, c, hi = decidecade_bands(100, 10000)
        # base-10 ratio 10^(1/10)
        assert c[1] / c[0] == pytest.approx(10 ** 0.1, rel=1e-6)
        # the 1 kHz band has the ISO nominal edges 891-1122 Hz
        i = int(np.argmin(np.abs(c - 1000)))
        assert lo[i] == pytest.approx(891.25, rel=1e-3)
        assert hi[i] == pytest.approx(1122.0, rel=1e-3)

    def test_flat_psd_band_levels_are_exactly_psd_times_bandwidth(self):
        """A band level is the PSD integrated over the WHOLE band ``[lo, hi]``,
        edges included. Integrating only the interior grid points under-reports
        by up to 2.6 dB on the low bands of this grid, and a band-to-band
        *difference* test cannot see it."""
        from uacpy.acoustic_signal.bands import (
            decidecade_bands, decidecade_band_levels,
        )
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER as REF
        f = np.linspace(1, 20000, 40000)
        psd = np.full_like(f, 1e-12)
        c, lv = decidecade_band_levels(psd, frequencies=f)
        lo, _, hi = decidecade_bands(f.min(), f.max())
        exact = 10 * np.log10(1e-12 * (hi - lo) / REF ** 2)
        covered = (lo >= f.min()) & (hi <= f.max()) & np.isfinite(lv)
        assert covered.sum() > 30
        np.testing.assert_allclose(lv[covered], exact[covered], atol=0.01)

    def test_white_noise_band_levels_rise_1db_per_band(self):
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.linspace(1, 20000, 40000)
        psd = np.ones_like(f) * 1e-12               # flat Pa²/Hz
        c, lv = decidecade_band_levels(psd, frequencies=f)
        step = np.diff(lv[(c > 200) & (c < 5000)])
        assert np.allclose(step, 1.0, atol=0.05)    # each band 10^0.1 wider -> +1 dB

    def test_bands_validate_input(self):
        from uacpy.acoustic_signal.bands import decidecade_bands
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError,
                           match='need 0 < freq_min < freq_max'):
            decidecade_bands(1000, 100)

    def test_coarse_grid_falls_back_not_nan(self):
        # A coarse, log-spaced grid leaves low bands with one sample; instead of
        # a silent NaN they get a rectangular estimate and a single warning.
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.logspace(np.log10(10), np.log10(2000), 25)
        psd = np.ones_like(f) * 1e-12
        with pytest.warns(UserWarning, match="too coarse"):
            c, lv = decidecade_band_levels(psd, frequencies=f)
        # coarse-grid bands still resolve a finite level
        assert np.any(np.isfinite(lv[c < 100]))


def test_acoustic_signal_is_importable():
    import uacpy.acoustic_signal as sig
    import uacpy
    assert sig is uacpy.acoustic_signal


def test_signal_symbols_resolve():
    import uacpy
    for name in ('lfm_chirp', 'hfm_chirp', 'tone_burst', 'gaussian_pulse',
                 'ricker_wavelet', 'make_bandlimited_noise',
                 'synthesize_time_series',
                 'welch', 'welch',
                 'probabilistic_welch',
                 'probabilistic_welch',
                 'sound_exposure', 'constant_q', 'spectrogram',
                 'SpectrogramResult', 'CWTResult', 'WignerVilleResult',
                 'FKResult', 'TauPResult', 'RadonResult'):
        assert hasattr(uacpy.acoustic_signal, name), \
            f"uacpy.acoustic_signal.{name} missing"


class TestSEL:
    """SEL must integrate power (Parseval-exact), not over-read the way a
    coherent-normalization Hann taper would (+1.76 dB on stationary signals,
    and an impulse at a segment boundary annihilated)."""

    def test_tone_exposure_is_parseval_exact(self):
        fs = 48000
        t = np.arange(fs) / fs
        x = 2.0 * np.sin(2 * np.pi * 1000.0 * t)   # exposure = A^2/2 * T = 2.0
        sel = _band_exposure(x, fs, band_type='octave', freq_min=10,
                             freq_max=20000, nperseg=fs).power
        assert sel.sum() == pytest.approx(np.sum(x ** 2) / fs, rel=1e-6)

    def test_impulse_not_annihilated(self):
        fs = 48000
        imp = np.zeros(fs)
        imp[0] = 10.0   # a Hann-windowed single segment would zero this out
        sel = _band_exposure(imp, fs, band_type='linear', freq_min=1.0,
                             freq_max=fs / 2, n_bands=240, nperseg=fs).power
        # full-band exposure ≈ Σx²/fs (only the excluded DC bin is dropped)
        assert sel.sum() == pytest.approx(np.sum(imp ** 2) / fs, rel=1e-3)

    def test_coarse_bands_do_not_double_count_bins(self):
        # 1-Hz FFT bins (nfft=fs) against the narrow low decidecade bands:
        # each bin must contribute to exactly one band, so a flat tone's total
        # exposure is conserved (no bin double-counted across overlapping bands).
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 50.0 * t)
        sel = _band_exposure(x, fs, band_type='decidecade', freq_min=8.9125,
                             freq_max=400, nperseg=fs).power
        assert sel.sum() == pytest.approx(np.sum(x ** 2) / fs, rel=1e-6)


def test_degenerate_input_guards_raise_configurationerror():
    """Pre-production robustness: degenerate inputs raise a typed
    ConfigurationError, not a raw ValueError/ZeroDivisionError."""
    from uacpy.core.exceptions import ConfigurationError
    from uacpy.acoustic_signal import cwt, tone_burst
    with pytest.raises(ConfigurationError,
                       match='data is empty'):       # sel: empty data
        _band_exposure(np.array([]), 48000.0)
    with pytest.raises(ConfigurationError,
                       match='integration_time must be > 0 s'):       # sel: zero integration_time
        _band_exposure(np.ones(2000), 48000.0, integration_time=0.0)
    with pytest.raises(ConfigurationError,
                       match='signal too short'):       # cwt: signal too short (n<8)
        cwt(np.ones(5), 8000.0)
    with pytest.raises(ConfigurationError,
                       match='frequency must be > 0 Hz'):       # tone_burst: frequency 0
        tone_burst(0.0, 5, sample_rate=1000.0)


class TestBandLimitedNoiseLandsInTheRequestedBand:
    """``scipy.signal.butter`` requires only ``0 < Wn < 1``. Clamping the
    normalised edges to 0.01/0.02 instead moved any low-frequency band at a high
    sample rate — the clamp is in *normalised* frequency, so its physical value
    scales with ``sample_rate`` and the same request is honoured at one rate and
    relocated at another.

    Removing the clamps alone is not enough: a narrow band at a high sample rate
    sits near a normalised frequency of 1e-3, where the transfer-function form
    loses so much precision the response collapses. Second-order sections are
    what make the requested band realisable, so both parts are exercised here.

    Realisations are seeded: the in-band fraction of a single draw carries real
    scatter (+/-0.05 at the hardest setting), so an unseeded threshold would be
    flaky rather than discriminating.
    """

    FS = 48_000.0

    @staticmethod
    def _spectrum(fc, bw, fs, seed):
        _t, n = make_bandlimited_noise(fc, bw, 2.0, sample_rate=fs,
                                       rng=np.random.default_rng(seed))
        freqs = np.fft.rfftfreq(n.size, 1.0 / fs)
        power = np.abs(np.fft.rfft(n)) ** 2
        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        return float(power[in_band].sum() / power.sum()), \
            float(freqs[np.argmax(power)])

    @pytest.mark.parametrize('fc,bw', [(100.0, 100.0), (250.0, 100.0),
                                       (1000.0, 200.0), (10_000.0, 10_000.0)])
    def test_most_power_lands_inside_the_request(self, fc, bw):
        """Pre-fix, the 50-150 Hz request delivered 0.1 % of its power there."""
        frac, peak = self._spectrum(fc, bw, self.FS, seed=0)
        assert frac > 0.8, f"only {frac:.1%} of power inside {fc}+/-{bw / 2}"
        assert fc - bw / 2 <= peak <= fc + bw / 2

    def test_the_request_is_honoured_at_both_sample_rates(self):
        """A clamp in *normalised* frequency makes the answer depend on the
        sample rate: pre-fix this band was correct at 4 kHz and relocated to
        252-457 Hz at 48 kHz. Both must now honour it, though the high-rate
        design remains the harder one."""
        for fs in (48_000.0, 4_000.0):
            frac, peak = self._spectrum(100.0, 100.0, fs, seed=0)
            assert frac > 0.8, f"{frac:.1%} in band at fs={fs:g}"
            assert 50.0 <= peak <= 150.0

    def test_the_requested_level_is_realised_in_band_in_pa(self):
        """``psd_level_dB`` is the in-band density in dB re 1 µPa²/Hz of a
        record in Pa. A design relocated in normalised frequency read 5.2 dB
        against a requested 40 dB and was internally self-consistent, so
        only a measured level catches it."""
        fs, fc, bw, level = self.FS, 100.0, 100.0, 40.0
        _, y = make_bandlimited_noise(fc, bw, 2.0, sample_rate=fs, psd_level_dB=level,
                                      rng=np.random.default_rng(0))
        freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
        psd = np.abs(np.fft.rfft(y)) ** 2 * 2.0 / (fs * y.size)
        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        assert 10 * np.log10(psd[in_band].mean() / 1e-12) == pytest.approx(
            level, abs=3.0)

    @pytest.mark.parametrize('fc,bw,fs', [(10.0, 100.0, 48_000.0),
                                          (23_990.0, 100.0, 48_000.0),
                                          (100.0, 100.0, 150.0)])
    def test_an_unrealisable_band_is_refused_not_moved(self, fc, bw, fs):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError,
                           match='make_bandlimited_noise: band .* not realisable'):
            make_bandlimited_noise(fc, bw, 0.5, sample_rate=fs)
        with pytest.raises(ConfigurationError,
                           match='make_bandlimited_noise: band .* not realisable'):
            make_bandlimited_noise(fc, bw, 0.5, sample_rate=fs, psd_level_dB=40.0)

    def test_the_package_dir_lists_public_names_and_no_loader_internals(self):
        """``dir(uacpy.acoustic_signal)`` is the index a user tab-completes:
        every public name, and not the stdlib ``importlib`` or the loader's
        private tables; the sub-modules stay importable as attributes and
        out of ``__all__``."""
        import types
        import uacpy.acoustic_signal as sig
        listed = set(dir(sig))
        assert set(sig.__all__) <= listed
        assert not {'importlib', '_importlib', '_EXPORTS', '_SUBMODULES',
                    '_SUBMODULE_NAMES'} & listed
        for name in sig._SUBMODULE_NAMES:
            assert name not in sig.__all__
            assert isinstance(getattr(sig, name), types.ModuleType)

    def test_duration_noise_generators_match_the_chirp_sample_count(self):
        """For ``duration = n/fs`` every duration-based generator returns the
        ``n`` samples ``lfm_chirp`` returns; ``int(duration*fs)`` is ``n - 1``
        for about 8 % of lengths at 48 kHz, and ``signal + noise`` then
        raises a broadcast error."""
        from uacpy.acoustic_signal.generate import (
            lfm_chirp, synthesize_noise_from_psd)
        fs = 48_000.0
        # The lengths where int((n/fs)*fs) == n - 1, found rather than listed.
        short = [n for n in range(1000, 2000) if int((n / fs) * fs) == n - 1]
        assert len(short) >= 10
        f = np.logspace(1, 3, 32)
        for n in short[:10]:
            dur = n / fs
            assert lfm_chirp(1000.0, 2000.0, dur, sample_rate=fs)[1].size == n
            assert make_bandlimited_noise(5000.0, 1000.0, dur, sample_rate=fs,
                                          rng=np.random.default_rng(0)
                                          )[1].size == n
            assert synthesize_noise_from_psd(
                1e-6 / (1 + (f / 100) ** 2), f, duration=dur, nfft=1024,
                sample_rate=fs, rng=np.random.default_rng(0))[1].size == n

    @pytest.mark.parametrize('fc,bw,fs', [(12_000.0, 5.0, 96_000.0),
                                          (12_002.9, 5.0, 96_000.0),
                                          (3_000.0, 2.0, 96_000.0)])
    def test_the_neb_of_a_narrow_band_matches_a_converged_grid(self, fc, bw, fs):
        """The noise-equivalent bandwidth sets the absolute level of
        ``make_bandlimited_noise(psd_level_dB=...)``, so a
        quadrature error in it is a level error of the same size in dB.

        A uniform ``n_freq``-point grid over the whole ``[0, Nyquist]`` samples
        a ``bw/fs = 5e-5`` band with a handful of points and reads it wrong by
        an amount that depends on where the passband falls between two grid
        points: +1.16 dB at ``fc=12000`` Hz and +3.65 dB at ``fc=12002.9`` Hz
        for the same 5 Hz band, +5.14 dB at 2 Hz. A band-focused grid of the
        same size is converged.
        """
        from scipy.signal import sosfreqz
        from uacpy.acoustic_signal.generate import (
            _bandpass_design, _noise_equivalent_bandwidth)

        sos = _bandpass_design(fc, bw, fs, who='test')
        # Reference: 2e6 points over fifty bandwidths of skirt each side.
        w = np.linspace(max(0.0, fc - bw / 2 - 50 * bw),
                        min(fs / 2, fc + bw / 2 + 50 * bw), 2_000_001)
        f, h = sosfreqz(sos, worN=w, fs=fs)
        power = np.abs(h) ** 4
        reference = float(np.trapezoid(power, f) / power.max())

        neb = _noise_equivalent_bandwidth(sos, fs, fc, bw)
        error_dB = 10 * np.log10(neb / reference)
        assert abs(error_dB) < 0.05, (
            f"NEB of the {bw:g} Hz band at fc={fc:g} Hz, fs={fs:g} Hz is "
            f"{neb:.6g} Hz against a converged {reference:.6g} Hz: "
            f"{error_dB:+.3f} dB, which ``psd_level_dB`` turns into the same "
            f"level error")

    def test_a_wide_band_neb_is_unchanged_by_the_focused_grid(self):
        """The focused window must not cost accuracy where the full-band grid
        was already converged: ten bandwidths of skirt puts the window edge
        below 1e-18 of the peak, so the two agree to rounding."""
        from scipy.signal import sosfreqz
        from uacpy.acoustic_signal.generate import (
            _bandpass_design, _noise_equivalent_bandwidth)

        fs, fc, bw = 96_000.0, 12_000.0, 1_000.0
        sos = _bandpass_design(fc, bw, fs, who='test')
        f, h = sosfreqz(sos, worN=8192, fs=fs)
        power = np.abs(h) ** 4
        full_band = float(np.trapezoid(power, f) / power.max())
        assert _noise_equivalent_bandwidth(sos, fs, fc, bw) == \
            pytest.approx(full_band, rel=1e-9)


class TestDecidecadePartialBandsAreNaN:
    """A band the supplied grid does not fully cover was returned as the
    integral over the *covered part*, which is not that band's level —
    measured 3.8 dB (first band) and 3.2 dB (last) off their own trend on a
    flat PSD, and 5.5 dB low on the realistic ``welch() -> band_levels`` path.
    The one warning the function emitted counted a different condition
    (bands with <2 interior grid points), so it fired for bands that were
    fine and stayed silent for the two that were wrong."""

    @staticmethod
    def _flat(freq_min=1.0, freq_max=25000.0, df=0.25):
        f = np.arange(freq_min, freq_max, df)
        return f, np.ones_like(f)

    def test_fully_covered_bands_are_exact_and_partial_ones_are_nan(self):
        from uacpy.acoustic_signal.bands import (
            decidecade_band_levels, decidecade_bands,
        )
        f, psd_flat = self._flat()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, levels = decidecade_band_levels(psd_flat, frequencies=f)
        lo, _, hi = decidecade_bands(f.min(), f.max())
        covered = (lo >= f.min()) & (hi <= f.max())
        exact = 10.0 * np.log10((hi - lo) / 1e-6 ** 2)
        # The discriminating half: the covered bands must stay exact, so the
        # fix cannot have been "widen a tolerance".
        assert np.nanmax(np.abs(levels[covered] - exact[covered])) < 1e-9
        assert np.all(np.isnan(levels[~covered]))

    def test_the_structural_end_bands_are_nan_and_are_not_warned_about(self):
        """The band set keeps every band *overlapping* the support, so the
        first and last are partial on any grid whose ends do not land on
        decidecade band edges — which no rfftfreq grid does. Their ``nan``
        level is the diagnostic; a warning about them fires on every
        well-formed call and cannot distinguish a short grid from a call."""
        from uacpy.acoustic_signal.bands import (
            decidecade_band_levels, decidecade_bands,
        )
        f, psd_flat = self._flat()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, levels = decidecade_band_levels(psd_flat, frequencies=f)
        assert not any('extend past' in str(c.message) for c in caught)
        lo, _, hi = decidecade_bands(f.min(), f.max())
        partial = (lo < f.min()) | (hi > f.max())
        assert partial.sum() in (1, 2)
        assert np.all(np.isnan(levels[partial]))

    def test_the_coarse_grid_warning_fires(self):
        """The negative control: the warning that qualifies *finite* levels is
        left in place."""
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.logspace(np.log10(10), np.log10(2000), 25)
        with pytest.warns(UserWarning, match='too coarse'):
            decidecade_band_levels(np.ones_like(f) * 1e-12, frequencies=f)

    def test_arrays_stay_parallel_with_decidecade_bands(self):
        # Shape contract: callers index the levels against a separately
        # computed decidecade_bands() with one mask, so dropping unsupported
        # bands would break them. nan keeps the arrays the same length.
        from uacpy.acoustic_signal.bands import (
            decidecade_band_levels, decidecade_bands,
        )
        f, psd_flat = self._flat()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centres, levels = decidecade_band_levels(psd_flat, frequencies=f)
        lo, ctr, hi = decidecade_bands(f.min(), f.max())
        assert centres.shape == levels.shape == ctr.shape


class TestWaveformDegenerateInputs:
    """Every generator raises a typed ConfigurationError on degenerate
    parameters and accepts plain lists for time vectors."""

    def test_hfm_chirp_degenerate_parameters_raise(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError,
                           match='freq_start and freq_end must differ'):   # equal bounds divide by zero
            hfm_chirp(1000.0, 1000.0, 0.1, sample_rate=8000.0)
        with pytest.raises(ConfigurationError,
                           match='freq_start must be > 0 Hz'):   # zero lower bound
            hfm_chirp(0.0, 500.0, 0.1, sample_rate=8000.0)
        with pytest.raises(ConfigurationError,
                           match='freq_end must be > 0 Hz'):   # zero upper bound
            hfm_chirp(500.0, 0.0, 0.1, sample_rate=8000.0)
        with pytest.raises(ConfigurationError,
                           match='duration must be > 0 s'):   # non-positive duration
            hfm_chirp(100.0, 500.0, 0.0, sample_rate=8000.0)

    def test_hfm_chirp_down_sweep_allowed(self):
        t, s = hfm_chirp(2000.0, 100.0, 0.1, sample_rate=8000.0)
        assert len(t) == len(s) > 0 and np.all(np.isfinite(s))

    def test_chirps_raise_instead_of_returning_empty(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='duration must be > 0 s'):
            lfm_chirp(100.0, 500.0, -1.0, sample_rate=8000.0)
        with pytest.raises(ConfigurationError,
                           match='must cover at least one sample'):   # under one sample long
            lfm_chirp(100.0, 500.0, 1e-6, sample_rate=8000.0)
        with pytest.raises(ConfigurationError, match='n_cycles must be > 0'):
            tone_burst(100.0, 0, sample_rate=8000.0)

    def test_lfm_equal_bounds_is_a_pure_tone(self):
        t, s = lfm_chirp(500.0, 500.0, 0.1, sample_rate=8000.0)
        np.testing.assert_allclose(s, np.sin(2 * np.pi * 500.0 * t))

    def test_pulses_reject_degenerate_parameters(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.generate import nwave, sparc_pulse
        time = np.linspace(0.0, 0.1, 64)
        with pytest.raises(ConfigurationError,
                           match='nwave: frequency must be > 0'):
            nwave(time, 0.0)
        with pytest.raises(ConfigurationError,
                           match='gaussian_pulse: duration must be > 0'):
            gaussian_pulse(time, 0.05, 0.0)
        with pytest.raises(ConfigurationError,
                           match='ricker_wavelet: frequency must be > 0'):
            ricker_wavelet(time, 0.0)
        with pytest.raises(ConfigurationError,
                           match='sparc_pulse: frequency must be > 0'):
            sparc_pulse(time, 0.0, "R")

    def test_the_generators_use_the_shared_signal_layer_scalar_guard(self):
        """Every waveform scalar goes through
        ``_validate.require_positive_finite_scalar`` — the same guard
        the other ``acoustic_signal`` modules apply — so the message names the
        parameter's unit and a non-finite value is refused, not only a
        non-positive one. A private copy of the check inside ``generate``
        gives neither: ``inf`` passes ``value > 0`` and produces silent
        garbage (an all-zero ``nwave``, a NaN chirp)."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.generate import nwave, sparc_pulse
        time = np.linspace(0.0, 0.1, 64)
        for call, pattern in (
                (lambda: sparc_pulse(time, np.inf, "R"),
                 r"sparc_pulse: frequency must be > 0 Hz and finite"),
                (lambda: ricker_wavelet(time, np.inf),
                 r"ricker_wavelet: frequency must be > 0 Hz and finite"),
                (lambda: gaussian_pulse(time, 0.05, np.nan),
                 r"gaussian_pulse: duration must be > 0 s and finite"),
                (lambda: lfm_chirp(100.0, 500.0, np.inf, sample_rate=8000.0),
                 r"lfm_chirp: duration must be > 0 s and finite"),
                (lambda: tone_burst(100.0, np.inf, sample_rate=8000.0),
                 r"tone_burst: n_cycles must be > 0 cycles and finite"),
                (lambda: hfm_chirp(100.0, np.inf, 0.1, sample_rate=8000.0),
                 r"hfm_chirp: freq_end must be > 0 Hz and finite"),
                (lambda: nwave(time, np.inf),
                 r"nwave: frequency must be > 0 Hz and finite"),
        ):
            with pytest.raises(ConfigurationError, match=pattern):
                call()

    def test_time_vector_functions_accept_lists(self):
        from uacpy.acoustic_signal.generate import nwave, sparc_pulse
        tl = [0.0, 0.001, 0.002, 0.005]
        assert ricker_wavelet(tl, 100.0).shape == (4,)
        assert gaussian_pulse(tl, 0.002, 0.001).shape == (4,)
        assert nwave(tl, 100.0).shape == (4,)
        assert sparc_pulse(tl, 100.0, "R")[0].shape == (4,)


def test_synthesize_noise_returns_the_rate_the_time_axis_uses():
    """The returned sample rate is the float rate the time axis was built
    from, also when the default 2*frequencies[-1] is not an integer."""
    from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
    f = np.array([1.0, 10.3])
    t, x, fs = synthesize_noise_from_psd(
        np.array([1e-6, 1e-6]), f, duration=0.5,
        rng=np.random.default_rng(0))
    assert isinstance(fs, float) and fs == 2 * 10.3
    assert abs(1.0 / (t[1] - t[0]) - fs) < 1e-9


class TestSynthesizeNoiseNamesTheBandAboveNyquist:
    """A flat 1e-4 Pa²/Hz target over 10 Hz - 20 kHz synthesised at 8 kHz
    keeps only the band up to 4 kHz: 0.399 Pa² of the 1.999 asked for."""

    f = np.array([10.0, 20000.0])
    p = np.array([1e-4, 1e-4])

    def test_a_target_straddling_nyquist_warns_with_the_fraction_lost(self):
        from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
        with pytest.warns(UserWarning, match=r"\(80 % of the band power\)"):
            _, x, _ = synthesize_noise_from_psd(
                self.p, self.f, duration=4.0, sample_rate=8000.0,
                rng=np.random.default_rng(0))
        assert np.var(x) == pytest.approx(1e-4 * 3990.0, rel=0.05)

    def test_a_target_ending_at_nyquist_is_silent(self):
        from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            synthesize_noise_from_psd(self.p, np.array([10.0, 4000.0]),
                                      duration=0.5, sample_rate=8000.0,
                                      rng=np.random.default_rng(0))

    def test_a_target_entirely_above_nyquist_raises(self):
        from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
        with pytest.raises(ConfigurationError, match="entirely at or above"):
            synthesize_noise_from_psd(self.p, np.array([4000.0, 20000.0]),
                                      sample_rate=8000.0)

    @pytest.mark.parametrize("rate", [-8000.0, 0.0, np.nan])
    def test_a_bad_sample_rate_is_named(self, rate):
        from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
        with pytest.raises(ConfigurationError, match="sample_rate must be"):
            synthesize_noise_from_psd(self.p, self.f, sample_rate=rate)


def test_m_sequence_polarity_and_autocorrelation():
    """The standard BPSK mapping s = 1 - 2*bit (bit 0 -> +1, bit 1 -> -1): a
    full period sums to -1 (2**(m-1) ones map to -1), and despreading with
    the same code keeps the symbol sign."""
    from uacpy.acoustic_signal.generate import m_sequence
    from uacpy.comms.coding import spread, despread
    s = m_sequence(5)
    d = m_sequence(5, [5, 3])
    assert s.sum() == -1 and d.sum() == -1
    # two-valued periodic autocorrelation survives the mapping
    ac = np.array([np.dot(s, np.roll(s, k)) for k in range(1, 31)])
    assert np.all(ac == -1)
    syms = np.array([1.0, -1.0, 1.0])
    rec = despread(spread(syms, s), s)
    np.testing.assert_allclose(rec.real, syms, atol=1e-12)


def test_more_degenerate_inputs_raise_typed_errors():
    """Degenerate parameters raise ConfigurationError, not ZeroDivisionError /
    ValueError / silently empty output."""
    from uacpy.core.exceptions import ConfigurationError
    from uacpy.acoustic_signal.generate import bpsk_modulate
    from uacpy.acoustic_signal.generate import synthesize_noise_from_psd
    from uacpy.acoustic_signal.frf import lsfir
    with pytest.raises(ConfigurationError,
                       match='chips_per_sec must be > 0'):       # chip rate of zero
        bpsk_modulate(np.array([1, -1]), 100.0, sample_rate=1000.0, chips_per_sec=0.0)
    with pytest.raises(ConfigurationError,
                       match='must cover at least one sample'):       # zero-length realisation
        synthesize_noise_from_psd(np.ones(8), np.linspace(10, 100, 8),
                                  duration=0)
    with pytest.raises(ConfigurationError,
                       match='is at or above the Nyquist frequency'):       # tone at/above Nyquist
        tone_burst(1000.0, 1, sample_rate=400.0)
    rng = np.random.default_rng(5)
    u = rng.standard_normal(64)
    y = np.convolve(u, [1.0, 0.5], mode="full")[:64]
    with pytest.raises(ConfigurationError,
                       match=r"FIR order \(100\) must be <= n_samples"):
        lsfir(u, y, 1000.0, order=100)


class TestSparcPulseLibraryShapes:
    """The 11-letter SPARC pulse library accepts every documented code and
    gates each pulse as ``cans.m`` does."""

    @pytest.mark.parametrize('code', list('PRASHNMGTCE'))
    def test_all_eleven_shapes_accepted(self, code):
        from uacpy.acoustic_signal.generate import sparc_pulse
        t = np.linspace(-0.05, 0.1, 512)
        s, title = sparc_pulse(t, 100.0, code)
        assert s.shape == t.shape
        assert np.all(np.isfinite(s))
        assert isinstance(title, str) and title
        assert np.any(s != 0)
        # Every shape but the sinc is gated to t > 0 (the sinc is the one
        # documented two-sided pulse).
        if code != 'C':
            assert np.all(s[t < 0] == 0)

    def test_unknown_code_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.generate import sparc_pulse
        with pytest.raises(ConfigurationError, match='Unknown pulse type'):
            sparc_pulse(np.linspace(0, 0.1, 64), 100.0, 'Z')

    def test_nwave_is_gated_to_one_period(self):
        from uacpy.acoustic_signal.generate import nwave
        f = 100.0
        t = np.linspace(-0.005, 0.02, 1001)
        s = nwave(t, f)
        outside = (t < 0) | (t > 1.0 / f)
        assert np.all(s[outside] == 0)
        inside = (t > 0) & (t < 1.0 / f)
        w = 2 * np.pi * f
        np.testing.assert_allclose(
            s[inside],
            np.sin(w * t[inside]) - 0.5 * np.sin(2 * w * t[inside]),
            atol=1e-12)


class TestMSequenceBounds:
    """``m_sequence`` register bounds and chip alphabet."""

    def test_m_sequence_rejects_an_out_of_range_register(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.generate import m_sequence
        for bad in (0, 1, -3, 2.5):
            with pytest.raises(ConfigurationError, match='integer >= 2'):
                m_sequence(bad)
        # The preset table ends at 15; a longer register needs its taps.
        with pytest.raises(ConfigurationError, match='preset table covers'):
            m_sequence(16)
        assert m_sequence(16, [16, 15, 13, 4]).size == 2 ** 16 - 1

    def test_m_sequence_chips_are_plus_minus_one_with_full_length(self):
        from uacpy.acoustic_signal.generate import m_sequence
        for m in (2, 7, 15):
            s = m_sequence(m)
            assert len(s) == 2 ** m - 1
            assert set(np.unique(s)) == {-1.0, 1.0}


class TestMakeMseqProbe:
    """Structure of the channel-sounding probe (docs/guide/signal.md §8):
    0.2 s zero leader, whole BPSK'd ``m_sequence(10)`` periods at chip rate
    ``(freq_max − freq_min)/2`` on carrier ``(freq_min + freq_max)/2``, normalised to 0.95
    full scale and zero-filled to exactly ``round(duration·fs)`` samples."""

    FS = 10_000.0
    FMIN, FMAX = 1000.0, 2000.0    # fc = 1500 Hz, 500 chips/s → 20 smp/chip
    N_PERIOD = 1023 * 20           # m_sequence(10) → 1023 chips

    def _probe(self, duration=5.0):
        from uacpy.acoustic_signal.generate import make_mseq_probe
        return make_mseq_probe(self.FMIN, self.FMAX, sample_rate=self.FS, duration=duration)

    def test_length_is_exactly_the_requested_duration(self):
        assert self._probe().size == 50_000     # round(5.0 * 10 kHz)

    def test_leader_is_zero_and_peak_is_095_full_scale(self):
        probe = self._probe()
        assert np.all(probe[:int(0.2 * self.FS)] == 0.0)
        assert np.max(np.abs(probe)) == pytest.approx(0.95)

    def test_whole_periods_repeat_and_tail_is_zero_filled(self):
        # 50 000 − 2 000 leader samples fit exactly two 20 460-sample
        # periods; a period is never truncated.
        probe = self._probe()
        lead = int(0.2 * self.FS)
        seg1 = probe[lead:lead + self.N_PERIOD]
        seg2 = probe[lead + self.N_PERIOD:lead + 2 * self.N_PERIOD]
        np.testing.assert_array_equal(seg1, seg2)
        assert np.any(seg1 != 0)
        assert np.all(probe[lead + 2 * self.N_PERIOD:] == 0.0)

    def test_probe_power_occupies_the_requested_band(self):
        probe = self._probe()
        freqs = np.fft.rfftfreq(probe.size, 1.0 / self.FS)
        power = np.abs(np.fft.rfft(probe)) ** 2
        in_band = (freqs >= self.FMIN) & (freqs <= self.FMAX)
        # Chip rate (freq_max − freq_min)/2 puts the BPSK main lobe (first sinc
        # nulls) exactly on [freq_min, freq_max]; measured in-band fraction 0.91.
        assert power[in_band].sum() / power.sum() > 0.85
        assert self.FMIN <= freqs[int(np.argmax(power))] <= self.FMAX

    def test_too_short_for_leader_plus_one_period_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        # One period lasts 2.046 s, so 1 s cannot hold leader + period.
        with pytest.raises(ConfigurationError, match='too short'):
            self._probe(duration=1.0)


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
        t, p = synthesize_time_series(amp * np.exp(-2j * np.pi * f * tau), frequencies=f,
                                      source_waveform=s, sample_rate=self.FS)
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
        trace = field.synthesize_time_series(s, self.FS, t_start=0.0)
        t, p = synthesize_time_series(H, frequencies=f, source_waveform=s, sample_rate=self.FS)
        assert np.array_equal(trace.coords['time'], t)
        assert np.array_equal(trace.data[0, 0], p)

    def test_the_frequency_axis_can_be_any_axis(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        H = np.exp(-2j * np.pi * f * 0.3)
        _, one = synthesize_time_series(H, frequencies=f, source_waveform=s, sample_rate=self.FS)
        _, last = synthesize_time_series(np.stack([H, 2 * H]), frequencies=f, source_waveform=s, sample_rate=self.FS)
        _, first = synthesize_time_series(np.stack([H, 2 * H]).T, frequencies=f, source_waveform=s,
                                          sample_rate=self.FS, axis=0)
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
            synthesize_time_series(np.ones(f.size, complex), frequencies=f, source_waveform=wf, sample_rate=self.FS)

    def test_a_mismatched_axis_is_refused(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        with pytest.raises(ConfigurationError, match='frequencies has'):
            synthesize_time_series(np.ones(f.size - 1, complex), frequencies=f, source_waveform=s,
                                   sample_rate=self.FS)

    def test_a_nan_bin_makes_the_trace_nan_and_warns(self):
        from uacpy.acoustic_signal import synthesize_time_series
        s, f = self._setup(25.0)
        H = np.ones((2, f.size), complex)
        H[1, 10] = np.nan
        with pytest.warns(UserWarning, match='1 of 2 cell'):
            _, p = synthesize_time_series(H, frequencies=f, source_waveform=s, sample_rate=self.FS)
        assert np.all(np.isfinite(p[0])) and np.all(np.isnan(p[1]))


class TestLfmChirpRefusesNegativeSweepBounds:
    @pytest.mark.parametrize("freq_start,freq_end", [(-500.0, 1000.0), (1000.0, -500.0),
                                           (NAN, 1000.0)])
    def test_a_bound_below_zero_raises(self, freq_start, freq_end):
        with pytest.raises(ConfigurationError, match="finite frequency >= 0"):
            lfm_chirp(freq_start, freq_end, 0.1, sample_rate=10000.0)

    def test_a_zero_start_frequency_is_accepted(self):
        _t, s = lfm_chirp(0.0, 1000.0, 0.1, sample_rate=10000.0)
        assert np.all(np.isfinite(s))


def _flat_psd():
    frequencies = np.logspace(1.0, 3.0, 16)
    return 1e-6 / (1.0 + (frequencies / 100.0) ** 2), frequencies


def test_small_nfft_warning_names_the_default_it_falls_back_to():
    """The behaviour is deliberate and documented: below 16 the argument is
    discarded for the 65536 default, not clamped up to 16. The message has to
    say that, or a reader takes it as ``nfft=16``."""
    Pxx, frequencies = _flat_psd()
    with pytest.warns(UserWarning,
                      match=r'below the minimum 16; using the default 65536'):
        _, small, _ = synthesize_noise_from_psd(
            Pxx, frequencies, duration=0.05, nfft=8, sample_rate=4000.0,
            rng=np.random.default_rng(0))
    _, default, _ = synthesize_noise_from_psd(
        Pxx, frequencies, duration=0.05, nfft=65536, sample_rate=4000.0,
        rng=np.random.default_rng(0))
    np.testing.assert_array_equal(small, default)


class TestBpskModulateBipolarChips:
    def test_zero_one_bits_raise_a_typed_error(self):
        with pytest.raises(ConfigurationError, match='chip'):
            bpsk_modulate(np.array([0, 1, 1, 0]), 100.0, sample_rate=1000.0, chips_per_sec=100.0)

    def test_bipolar_chips_emit_one_signed_carrier_block_per_chip(self):
        chips = np.array([1, -1, 1, 1, -1, 1])
        fc, fs, cps = 100.0, 1000.0, 100.0
        s = bpsk_modulate(chips, fc, sample_rate=fs, chips_per_sec=cps)
        tone = np.sin(2 * np.pi * fc * np.arange(int(fs / cps)) / fs)
        np.testing.assert_allclose(
            s, np.concatenate([c * tone for c in chips]), atol=1e-12)


class TestWelchMasksDegenerateDenominators:
    def test_zero_input_masks_h1_and_coherence_to_nan_with_warning(self):
        rng = np.random.default_rng(0)
        frf = FRF()
        with pytest.warns(UserWarning, match="frf_welch"):
            result = frf.compute(np.zeros(2048), rng.standard_normal(2048),
                                 1000.0, method="welch", nperseg=256)
        assert np.isnan(result.transfer_function).all()
        assert np.isnan(result.coherence).all()

    def test_zero_output_masks_h2_to_nan_with_warning(self):
        rng = np.random.default_rng(1)
        frf = FRF(estimator="H2")
        with pytest.warns(UserWarning, match="cross-spectral"):
            _, tf = frf.compute(rng.standard_normal(2048), np.zeros(2048),
                                1000.0, method="welch", nperseg=256)
        assert np.isnan(tf).all()

    def test_excited_records_give_finite_h1_h2_and_coherence_silently(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(4096)
        y = np.convolve(x, [1.0, 0.4])[:4096]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result1 = FRF().compute(x, y, 1000.0, method="welch",
                                    nperseg=512)
            _, tf2 = FRF(estimator="H2").compute(x, y, 1000.0,
                                                 method="welch", nperseg=512)
        assert np.isfinite(result1.transfer_function).all()
        assert np.isfinite(tf2).all()
        assert np.isfinite(result1.coherence).all()


class TestSampleRateGuards:
    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_psd_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="welch: sample_rate must "
                                 "be > 0 Hz and finite"):
            welch(np.ones(64), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_ppsd_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="probabilistic_welch: "
                                 "sample_rate must be > 0 Hz"):
            probabilistic_welch(np.ones(64), bad, segment_duration=0.1)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_pulse_compression_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="pulse_compression: sample_rate"):
            pulse_compression(np.ones(32), np.ones(8), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_ambiguity_function_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="ambiguity_function: sample_rate"):
            ambiguity_function(np.ones(8), bad, n_doppler=3)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_instantaneous_frequency_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="instantaneous_frequency: sample_rate"):
            instantaneous_frequency(np.sin(0.3 * np.arange(32)), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_fk_transform_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="fk_transform: sample_rate"):
            fk_transform(np.zeros((16, 4)), bad, 1.0)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_fk_transform_rejects_bad_dx(self, bad):
        # A negative dx silently mirrored the wavenumber axis; zero divided
        # by it raw.
        with pytest.raises(ConfigurationError,
                           match="fk_transform: dx must be > 0 m and finite"):
            fk_transform(np.zeros((16, 4)), 100.0, bad)


class TestSampleRateFiniteness:
    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_spectrogram_rejects_nonpositive_or_nonfinite_rate(self, bad):
        # fs=inf passed the > 0 check and raised ZeroDivisionError in scipy.
        with pytest.raises(ConfigurationError,
                           match="spectrogram: sample_rate must be > 0 Hz "
                                 "and finite"):
            spectrogram(np.ones(256), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_band_exposure_rejects_nonpositive_or_nonfinite_rate(self, bad):
        # fs=inf reached nfft = int(sample_rate) and raised OverflowError.
        with pytest.raises(ConfigurationError,
                           match="sound_exposure: sample_rate must be > 0 "
                                 "Hz and finite"):
            _band_exposure(np.ones(256), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_cwt_rejects_nonpositive_or_nonfinite_rate(self, bad):
        # fs=inf with explicit frequencies produced inf scales and an
        # all-NaN coefficient matrix.
        with pytest.raises(ConfigurationError,
                           match="cwt: sample_rate must be > 0 Hz and "
                                 "finite"):
            cwt(np.ones(256), bad, frequencies=[10.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_bpsk_modulate_rejects_nonfinite_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="bpsk_modulate: sample_rate must be > 0 Hz "
                                 "and finite"):
            bpsk_modulate(np.array([1, -1]), 100.0, sample_rate=bad, chips_per_sec=100.0)

    def test_bpsk_modulate_rejects_infinite_chip_rate(self):
        # chips_per_sec=inf gave samples_per_chip = 0 and a silently empty
        # waveform.
        with pytest.raises(ConfigurationError,
                           match="bpsk_modulate: chips_per_sec must be > 0 "
                                 "chips/s and finite"):
            bpsk_modulate(np.array([1, -1]), 100.0, sample_rate=1000.0, chips_per_sec=np.inf)


class TestBandlimitedNoiseMatchesTheRecordLengthExactly:
    """``make_bandlimited_noise`` sizes its record as
    ``round(duration*sample_rate)``, so ``duration = n/fs`` gives ``n``.

    ``int((n/fs)*fs)`` is ``n - 1`` for 7.02 % of lengths at 44100 Hz, 6.14 %
    at 48000 Hz and 5.16 % at 9600 Hz — always one short, never one long — so
    a count routed through ``int`` produces a noise vector the record cannot
    be added to.
    """

    @pytest.mark.parametrize('n, fs', [
        (1000, 8000.0), (1001, 8000.0), (5008, 9600.0), (44100, 44100.0),
        (30011, 48000.0),
    ])
    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_the_output_has_the_record_length(self, n, fs, n_channels):
        _, out = make_bandlimited_noise(1000.0, 500.0, n / fs, sample_rate=fs,
                                        psd_level_dB=40.0,
                                        n_channels=n_channels,
                                        rng=np.random.default_rng(0))
        assert out.shape == ((n,) if n_channels == 1 else (n, n_channels))

    @pytest.mark.parametrize('n, fs', [(1001, 8000.0), (5008, 9600.0)])
    def test_the_lengths_this_covers_are_the_ones_int_would_lose(
            self, n, fs):
        assert int((n / fs) * fs) == n - 1

    def test_a_record_shorter_than_the_filter_padding_is_named(self):
        with pytest.raises(
                ConfigurationError,
                match='is too short for the zero-phase bandpass') as exc:
            make_bandlimited_noise(1000.0, 500.0, 20 / 8000.0, sample_rate=8000.0,
                                   psd_level_dB=40.0,
                                   rng=np.random.default_rng(0))
        message = str(exc.value)
        assert 'make_bandlimited_noise' in message
        assert '20 sample(s)' in message

    def test_the_duration_taking_entry_point_sizes_by_duration(self):
        _, noise = make_bandlimited_noise(1000.0, 500.0, 1.0, sample_rate=8000.0,
                                          rng=np.random.default_rng(0))
        assert noise.size == int(1.0 * 8000.0)


class TestSelRefusesANonPositiveIntegrationTime:
    """``data[:int(integration_time*fs)]`` is a Python end-slice for a
    negative value, so ``integration_time=-1.0`` on a 5 s record returns
    bit-identically what ``+4.0`` returns — a confident, plausible SEL for an
    input that is nonsense (a difference of timestamps taken the wrong way
    round). NaN and Inf reached ``int()`` and raised untyped errors."""

    @staticmethod
    def _record():
        return np.random.default_rng(0).standard_normal(5000), 1000.0

    @pytest.mark.parametrize('bad', [-1.0, -3.0, -10.0, 0.0, NAN, np.inf])
    def test_a_non_positive_or_non_finite_value_raises(self, bad):
        data, fs = self._record()
        with pytest.raises(ConfigurationError, match='integration_time'):
            _band_exposure(data, fs, integration_time=bad)

    @pytest.mark.parametrize('good, n_expected', [(4.0, 4000), (2.0, 2000)])
    def test_a_positive_value_truncates_from_the_start(
            self, good, n_expected):
        data, fs = self._record()
        got = np.nansum(_band_exposure(data, fs, integration_time=good).power)
        want = np.nansum(_band_exposure(data[:n_expected], fs).power)
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_smallest_admissible_value_is_one_sample_of_data(self):
        # The boundary the guard leaves alone: any value > 0 is accepted, and
        # the emptiness it can still produce is the other guard's message.
        data, fs = self._record()
        with pytest.raises(ConfigurationError, match='no samples to integrate'):
            _band_exposure(data, fs, integration_time=1e-9)


class TestNyquistGuardsSplitGeneratorsFromAnalysers:
    """Every entry point that takes a frequency and a sample rate answers the
    same question, and the two answers it may give are deliberate.

    Generators refuse ``f == fs/2``: two samples per cycle carry no phase, so
    a sinusoid there is degenerate. Analysers admit it, because the Nyquist
    bin is a real bin of an ``rfft`` grid and the default analysis grids cap
    themselves at exactly ``fs/2`` (``docs/guide/signal.md`` states this for
    ``cwt``). ``require_below_nyquist`` and ``require_at_most_nyquist`` are
    the two sides, so a new entry point picks one rather than forgetting.
    """

    FS = 10000.0

    @pytest.mark.parametrize('name, call', [
        ('tone_burst',
         lambda f, fs: tone_burst(f, 5, sample_rate=fs)),
        ('lfm_chirp',
         lambda f, fs: lfm_chirp(100.0, f, 0.1, sample_rate=fs)),
        ('hfm_chirp',
         lambda f, fs: hfm_chirp(100.0, f, 0.1, sample_rate=fs)),
        ('bpsk_modulate',
         lambda f, fs: bpsk_modulate(np.array([1, -1, 1, -1]), f, sample_rate=fs, chips_per_sec=100.0)),
        ('fsk_modulate',
         lambda f, fs: _fsk_modulate(np.array([0, 1]), np.array([1000.0, f]),
                                     0.01, fs)),
        ('fsk_demodulate',
         lambda f, fs: _fsk_demodulate(np.zeros(1000),
                                       np.array([1000.0, f]), 0.01, fs)),
    ])
    def test_a_generator_refuses_exactly_nyquist_and_accepts_just_below(
            self, name, call):
        fs = self.FS
        with pytest.raises(ConfigurationError, match='Nyquist'):
            call(fs / 2, fs)
        call(fs / 2 - 1.0, fs)

    @pytest.mark.parametrize('fc', [1000.0, 2000.0, 4000.0])
    def test_an_admitted_band_lands_where_it_was_asked_for(self, fc):
        """The negative half of the guard: below Nyquist the band is where the
        caller put it, so the guard is refusing folds and nothing else."""
        _, x = make_bandlimited_noise(fc, 200.0, 1.0, sample_rate=self.FS,
                                      rng=np.random.default_rng(0))
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.FS)
        peak = freqs[int(np.argmax(np.abs(np.fft.rfft(x))))]
        assert abs(peak - fc) < 150.0

    def test_the_analyser_side_admits_exactly_nyquist_and_refuses_above(self):
        from uacpy.acoustic_signal import constant_q
        fs = self.FS
        x = np.random.default_rng(0).standard_normal(8000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            constant_q(x, fs, scaling='spectrum', freq_min=100.0, freq_max=fs / 2)
        with pytest.raises(ConfigurationError, match='Nyquist'):
            constant_q(x, fs, scaling='spectrum', freq_min=100.0, freq_max=fs / 2 + 1.0)

    def test_the_two_helpers_disagree_only_at_exactly_nyquist(self):
        from uacpy.core._validate import (
            require_at_most_nyquist, require_below_nyquist,
        )
        fs = self.FS
        require_below_nyquist(fs / 2 - 1e-9, fs, 'probe', 'f', 'it aliases')
        require_at_most_nyquist(fs / 2, fs, 'probe', 'f', 'it aliases')
        with pytest.raises(ConfigurationError,
                           match='is at or above the Nyquist frequency'):
            require_below_nyquist(fs / 2, fs, 'probe', 'f', 'it aliases')
        with pytest.raises(ConfigurationError,
                           match='is above the Nyquist frequency'):
            require_at_most_nyquist(fs / 2 + 1e-9, fs, 'probe', 'f',
                                    'it aliases')

    def test_an_array_valued_guard_names_the_offending_frequencies(self):
        from uacpy.core._validate import require_below_nyquist
        with pytest.raises(
                ConfigurationError,
                match='are at or above the Nyquist frequency') as exc:
            require_below_nyquist(np.array([100.0, 9000.0, 7000.0]), self.FS,
                                  'probe', 'tone(s)', 'the tones alias')
        message = str(exc.value)
        named = message.split('tone(s) ')[1].split(' Hz are')[0]
        assert '7000' in named and '9000' in named
        assert '100' not in named


class TestPositiveScalarGuardsCoverEveryAcousticSignalEntryPoint:
    """Seven entry points divide by a sample rate, a sensor spacing, a range
    or a sound speed, and each is validated where 31 siblings in this package
    already are.

    ``wigner_ville`` is the quiet one: its ``distribution`` array comes back
    finite and correct at ``sample_rate = 0`` and the frequency and time axes
    the caller plots it against are the garbage.
    """

    DATA_1D = np.random.default_rng(0).standard_normal(256)
    DATA_2D = np.random.default_rng(0).standard_normal((64, 8))

    @pytest.mark.parametrize('bad', BAD_SCALARS)
    @pytest.mark.parametrize('name, call', [
        ('wigner_ville', lambda d, b: _wigner_ville(d, b)),
        ('constant_q',
         lambda d, b: _constant_q_spectrum(d, b, freq_min=20.0, freq_max=100.0)),
        ('warp_signal', lambda d, b: _warp_signal(d, b, 1000.0)),
    ])
    def test_a_bad_sample_rate_raises_a_typed_error(self, name, call, bad):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            call(self.DATA_1D, bad)

    @pytest.mark.parametrize('bad', BAD_SCALARS)
    @pytest.mark.parametrize('name, call', [
        ('taup_transform', lambda d, b: _taup_transform(d, b, 1.0)),
        ('radon_transform',
         lambda d, b: _radon_transform(d, b, 1.0, np.array([1500.0]))),
    ])
    def test_a_bad_sample_rate_on_a_gather_raises(self, name, call, bad):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            call(self.DATA_2D, bad)

    @pytest.mark.parametrize('bad', BAD_SCALARS)
    @pytest.mark.parametrize('name, call', [
        ('taup_transform', lambda d, b: _taup_transform(d, 1000.0, b)),
        ('radon_transform',
         lambda d, b: _radon_transform(d, 1000.0, b, np.array([1500.0]))),
    ])
    def test_a_bad_sensor_spacing_raises(self, name, call, bad):
        with pytest.raises(ConfigurationError, match='dx'):
            call(self.DATA_2D, bad)

    @pytest.mark.parametrize('bad', BAD_SCALARS)
    @pytest.mark.parametrize('arg', ['range_m', 'sound_speed'])
    def test_warp_signal_guards_its_range_and_sound_speed(self, arg, bad):
        kwargs = {'range_m': 1000.0}
        kwargs[arg] = bad
        with pytest.raises(ConfigurationError,
                           match=f"warp_signal: {arg} must be"):
            _warp_signal(self.DATA_1D, 1000.0, **kwargs)

    def test_wigner_ville_axes_are_finite_on_an_accepted_rate(self):
        """The negative control: a valid rate still returns usable axes, so
        the guard is refusing the garbage cases and nothing else."""
        result = _wigner_ville(self.DATA_1D, 1000.0)
        assert np.all(np.isfinite(result.frequencies))
        assert np.all(np.isfinite(result.times))
        assert np.all(np.diff(result.frequencies) > 0)


class TestScalarArgumentsAreRefusedByName:
    """NaN, non-positive and inverted scalar arguments come back as a typed
    error naming the argument, never as numpy's own ValueError or IndexError."""

    X = np.random.default_rng(0).standard_normal(4000)

    @pytest.mark.parametrize('bad', [np.nan, -1.0, 0.0])
    def test_make_bandlimited_noise_names_its_duration(self, bad):
        with pytest.raises(ConfigurationError, match='duration must be'):
            make_bandlimited_noise(1000.0, 200.0, bad, sample_rate=8000.0)

    def test_make_bandlimited_noise_names_its_sample_rate(self):
        with pytest.raises(ConfigurationError, match='sample_rate must be'):
            make_bandlimited_noise(1000.0, 200.0, 1.0, sample_rate=np.nan)

    @pytest.mark.parametrize('kwargs, name', [
        (dict(segment_duration=np.nan), 'segment_duration must be'),
        (dict(level_step_dB=np.nan), 'level_step_dB must be'),
        (dict(level_step_dB=0.0), 'level_step_dB must be'),
        (dict(level_min_dB=100, level_max_dB=50), 'level_min_dB < level_max_dB'),
        (dict(level_min_dB=50, level_max_dB=50), 'level_min_dB < level_max_dB'),
    ])
    def test_the_histogram_scalars_are_named(self, kwargs, name):
        from uacpy.acoustic_signal import probabilistic_welch
        with pytest.raises(ConfigurationError, match=name):
            probabilistic_welch(self.X, 2000.0, segment_duration=kwargs.pop(
                'segment_duration', 0.5), nperseg=256, **kwargs)

    def test_the_constant_q_histogram_takes_the_same_level_window_rule(self):
        from uacpy.acoustic_signal import probabilistic_constant_q
        with pytest.raises(ConfigurationError, match='level_min_dB < level_max_dB'):
            probabilistic_constant_q(self.X, 2000.0, freq_min=100.0,
                                     level_min_dB=100, level_max_dB=50)

    def test_a_level_window_one_bin_wide_is_accepted(self):
        from uacpy.acoustic_signal import probabilistic_welch
        r = probabilistic_welch(self.X, 2000.0, segment_duration=0.5,
                                nperseg=256, level_min_dB=50, level_max_dB=51)
        assert r.level_edges.tolist() == [50.0, 51.0]

    def test_uniform_frequency_step_needs_two_frequencies(self):
        from uacpy.acoustic_signal.channel import uniform_frequency_step
        with pytest.raises(ConfigurationError, match='at least two'):
            uniform_frequency_step([100.0])
        assert uniform_frequency_step([100.0, 110.0]) == pytest.approx(10.0)

    @pytest.mark.parametrize('method', ['welch', 'etfe', 'ls_fir', 'p_etfe'])
    def test_frf_refuses_records_of_different_lengths(self, method):
        with pytest.raises(ConfigurationError, match='same length'):
            FRF(method=method, order=4).compute(self.X[:1000], self.X[:900],
                                            1000.0)

    @pytest.mark.parametrize('bad', [0.0, -1000.0, np.nan])
    def test_frf_names_its_sample_rate(self, bad):
        with pytest.raises(ConfigurationError, match='sample_rate must be'):
            FRF(method='etfe').compute(self.X, self.X, bad)

    @pytest.mark.parametrize('bad', ['', None, 5])
    def test_sparc_pulse_refuses_a_pulse_type_that_names_no_letter(self, bad):
        from uacpy.acoustic_signal.generate import sparc_pulse
        with pytest.raises(ConfigurationError,
                           match='pulse_type must be a non-empty string'):
            sparc_pulse(np.linspace(0, 0.01, 10), 100.0, bad)

    def test_simulate_reception_matches_the_direct_convolution(self):
        """A long transmit through a long channel takes scipy's FFT route; the
        reception is the direct sum's to rounding."""
        from uacpy.acoustic_signal.channel import (
            impulse_response, simulate_reception,
        )
        fs = 48000.0
        x = np.random.default_rng(1).standard_normal(9600)
        amps, delays = [1.0, 0.5, 0.25], [0.01, 0.3, 1.0]
        _, y = simulate_reception(x, amps, delays, fs)
        _, h = impulse_response(amps, delays, sample_rate=fs)
        np.testing.assert_allclose(y, np.convolve(x, h), rtol=0, atol=1e-12)


class TestDspEstimatorsNameTheBridgeWhenHandedAField:
    """``acoustic_signal`` is array-in / array-out; a ``Field`` reaches
    ``np.isfinite`` as an object array and raises ``ufunc 'isfinite' not
    supported for the input types``, which names nothing the caller passed."""

    @staticmethod
    def _trace():
        from uacpy.core.results import Field
        t = np.arange(1600) / 1600.0
        return Field(data=np.sin(2 * np.pi * 100 * t), coords={'time': t})

    def test_psd_names_the_data_attribute(self):
        with pytest.raises(ConfigurationError,
                           match=r'Pass Field\.data') as exc:
            welch(self._trace(), 1600.0)
        message = str(exc.value)
        assert 'welch:' in message
        assert 'Field' in message
        assert 'Field.data' in message

    def test_the_bridge_the_message_names_actually_works(self):
        trace = self._trace()
        freqs, power = welch(np.asarray(trace.data), 1600.0)
        assert freqs.size == power.size


class TestSignalAxisGuardsRefuseAnEmptyAxis:
    """Three entry points validated axis monotonicity but not axis emptiness,
    so an empty frequency axis reached numpy and raised an untyped
    ``ValueError`` / ``IndexError`` naming no input the caller supplied — the
    exact failure the canonical guard in ``core._validate`` says it
    exists to prevent.

    The monotonicity predicate itself agrees across all five copies of it in
    the package; only the empty and one-sample corners differed.
    """

    @staticmethod
    def _calls():
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function,
        )
        from uacpy.acoustic_signal.dispersion import modal_group_velocity
        return {
            # A monotonic k_horizontal: a propagating mode's wavenumber
            # rises with frequency, and a flat one is refused (it divides the
            # group velocity by zero), so a constant axis would not exercise
            # the well-formed path this helper is shared with.
            'modal_group_velocity':
                lambda f, n: modal_group_velocity(f, k_horizontal=np.linspace(0.5, 12.0, n)),
            'impulse_response_from_transfer_function':
                lambda f, n: impulse_response_from_transfer_function(
                    np.ones(n, dtype=complex), frequencies=f, sample_rate=1000.0),
            'decidecade_band_levels':
                lambda f, n: decidecade_band_levels(np.ones(n), frequencies=f),
        }

    @pytest.mark.parametrize('name', ['modal_group_velocity',
                                      'impulse_response_from_transfer_function',
                                      'decidecade_band_levels'])
    def test_an_empty_axis_raises_a_typed_error_naming_the_axis(self, name):
        call = self._calls()[name]
        with pytest.raises(
                ConfigurationError,
                match='frequencies must contain at least one value') as exc:
            call(np.array([], dtype=float), 0)
        message = str(exc.value)
        assert name in message
        assert 'frequencies' in message
        assert 'at least one value' in message

    @pytest.mark.parametrize('name', ['modal_group_velocity',
                                      'decidecade_band_levels'])
    def test_a_one_sample_axis_names_its_own_domain_minimum(self, name):
        call = self._calls()[name]
        with pytest.raises(
                ConfigurationError,
                match='frequencies needs at least 2 samples') as exc:
            call(np.array([100.0]), 1)
        message = str(exc.value)
        assert name in message
        assert 'at least 2 samples' in message

    @pytest.mark.parametrize('name', ['modal_group_velocity',
                                      'impulse_response_from_transfer_function',
                                      'decidecade_band_levels'])
    def test_a_non_monotonic_axis_gets_the_domain_hint(self, name):
        """The negative control for the shared guard: it backstops the empty
        case without displacing the local message, which is where each
        function's own remediation lives."""
        call = self._calls()[name]
        with pytest.raises(ConfigurationError,
                           match='frequencies must be .*increasing') as exc:
            call(np.array([1.0, 3.0, 2.0, 4.0]), 4)
        message = str(exc.value)
        assert 'increasing' in message
        hints = {
            'modal_group_velocity': 'non-increasing step',
            'impulse_response_from_transfer_function': 'Sort the axis',
            'decidecade_band_levels': 'fftfreq',
        }
        assert hints[name] in message

    def test_synthesize_noise_from_psd_requires_two_points(self):
        """Deliberately NOT routed through the shared guard: the shared guard
        accepts a one-sample axis and this function documents a two-point
        minimum, so routing it would relax a stated requirement."""
        from uacpy.acoustic_signal.generate import (
            synthesize_noise_from_psd)
        for n in (0, 1):
            with pytest.raises(ConfigurationError, match='at least 2 points'):
                synthesize_noise_from_psd(np.ones(n),
                                          np.arange(n, dtype=float) + 1.0,
                                          sample_rate=1000.0)

    def test_a_well_formed_axis_runs_through_all_three(self):
        f = np.linspace(100.0, 4000.0, 64)
        calls = self._calls()
        assert calls['modal_group_velocity'](f, 64).shape == (64,)
        _t, h = calls['impulse_response_from_transfer_function'](f, 64)
        assert h.size > 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert calls['decidecade_band_levels'](f, 64) is not None


class TestOneMSequenceGenerator:
    """One generator serves the sonar probes and the spreading codes, with one
    seed convention, so a spread and a despread built from the same
    ``(n_register, taps)`` always line up, and the preset table reproduces
    the Acoustics-Toolbox ``mseq.m`` recursion chip for chip."""

    SYMBOLS = np.array([1.0, -1.0, 1.0])

    @staticmethod
    def _mseq_m(m):
        """``mseq.m``'s shift-register recursion, with this package's BPSK
        mapping (bit 0 -> +1): seed [1, 0, ..., 0], feedback dot(c, seed)."""
        from uacpy.acoustic_signal.generate import _MSEQ_FEEDBACK
        c = np.array(_MSEQ_FEEDBACK[m])
        seed = np.zeros(m)
        seed[0] = 1
        out = np.zeros(2 ** m - 1)
        for i in range(out.size):
            seed = np.concatenate([seed[1:], [np.mod(np.dot(c, seed), 2)]])
            out[i] = seed[0]
        return 1.0 - 2.0 * out

    @pytest.mark.parametrize('m', range(2, 16))
    def test_the_preset_table_reproduces_the_toolbox_recursion(self, m):
        from uacpy.acoustic_signal.generate import m_sequence
        np.testing.assert_array_equal(m_sequence(m), self._mseq_m(m))

    def test_explicit_taps_of_the_preset_polynomial_give_the_same_sequence(
            self):
        from uacpy.acoustic_signal.generate import m_sequence
        np.testing.assert_array_equal(m_sequence(5, [5, 2]), m_sequence(5))
        np.testing.assert_array_equal(m_sequence(7, [7, 1]), m_sequence(7))

    def test_comms_and_acoustic_signal_share_the_one_function(self):
        import uacpy.acoustic_signal as sig
        import uacpy.comms as comms
        assert comms.m_sequence is sig.m_sequence
        assert not hasattr(sig, 'mseq')
        from uacpy.comms import coding
        assert not hasattr(coding, 'm_sequence')

    @pytest.mark.parametrize('taps', [None, (5, 3)])
    def test_the_same_call_at_both_ends_recovers_the_symbols(self, taps):
        from uacpy.acoustic_signal.generate import m_sequence
        from uacpy.comms.coding import despread, spread
        seq = m_sequence(5, taps)
        got = despread(spread(self.SYMBOLS, seq), seq)
        np.testing.assert_allclose(np.real(got), self.SYMBOLS, atol=1e-9)

    def test_a_non_primitive_tap_set_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.generate import m_sequence
        with pytest.raises(ConfigurationError, match='not primitive'):
            m_sequence(5, [5, 1])


class TestEstimatorOutputDtypesMatchTheDocumentedSplit:
    """The package docstring states which estimators preserve a ``float32``
    input and which promote. Undocumented, the split surfaces as a silent
    promotion when two estimates of the same record are stacked."""

    FS = 1000.0

    @staticmethod
    def _record(dtype):
        return np.random.default_rng(0).standard_normal(4096).astype(dtype)

    @pytest.mark.parametrize('name, call, field', [
        ('psd', lambda x, fs: welch(x, fs), 'power'),
        ('spectrogram', lambda x, fs: spectrogram(x, fs), 'power'),
    ])
    def test_the_two_preserving_estimators_keep_float32(self, name, call,
                                                        field):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = call(self._record(np.float32), self.FS)
        assert np.asarray(getattr(result, field)).dtype == np.float32
        assert np.asarray(result.frequencies).dtype == np.float64

    @pytest.mark.parametrize('name, call', [
        ('envelope', lambda x, fs: _envelope(x)),
        ('cepstrum', lambda x, fs: _cepstrum(x)),
        ('constant-Q spectrum',
         lambda x, fs: _constant_q_spectrum(x, fs, freq_min=20.0, freq_max=400.0)),
        ('band exposure', lambda x, fs: _band_exposure(x, fs).power),
        ('wigner_ville', lambda x, fs: _wigner_ville(x, fs).distribution),
    ])
    def test_the_promoting_estimators_return_float64(self, name, call):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = call(self._record(np.float32), self.FS)
        arr = getattr(result, 'power', result)
        assert np.asarray(arr).dtype == np.float64

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_the_analytic_transforms_are_complex128_from_either_input(
            self, dtype):
        from uacpy.acoustic_signal import analytic_signal, fk_transform
        assert analytic_signal(self._record(dtype)).dtype == np.complex128
        gather = np.random.default_rng(0).standard_normal((256, 8)).astype(dtype)
        assert fk_transform(gather, self.FS, 1.0).spectrum.dtype == np.complex128

    def test_nothing_downcasts_a_float64_record(self):
        """The half that must not change: a float64 record stays float64
        everywhere, so the split is about promotion only."""
        x = self._record(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert welch(x, self.FS).power.dtype == np.float64
            assert spectrogram(x, self.FS).power.dtype == np.float64
            assert _envelope(x).dtype == np.float64


class TestRickerWaveletDelay:
    """``delay=`` places the wavelet's centre, where the default keeps the
    Acoustics-Toolbox offset. Its sibling ``gaussian_pulse`` always took a
    delay; this one did not, so a gather needing one pulse per trace at a
    moveout-dependent time had to reimplement the wavelet locally."""

    def test_the_default_centring_is_unchanged(self):
        """AT's ``u = 2πFt − 8`` puts the central lobe at 4/(pi*f)."""
        frequency = 40.0
        time = np.linspace(0, 0.2, 2001)

        centre = time[np.argmin(ricker_wavelet(time, frequency))]
        assert centre == pytest.approx(4 / (np.pi * frequency), abs=1e-4)

    def test_the_equivalent_delay_reproduces_the_default(self):
        """Algebraically the same expression, so the two agree to round-off —
        about 1e-15 of the lobe, not bit-identical."""
        frequency = 40.0
        time = np.linspace(0, 0.2, 2001)

        default = ricker_wavelet(time, frequency)
        explicit = ricker_wavelet(time, frequency,
                                  delay=4 / (np.pi * frequency))
        assert np.allclose(default, explicit, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize('delay', [0.05, 0.10, 0.15])
    def test_the_wavelet_lands_where_it_is_asked_to(self, delay):
        time = np.linspace(0, 0.2, 2001)

        wavelet = ricker_wavelet(time, 40.0, delay=delay)
        assert time[np.argmin(wavelet)] == pytest.approx(delay, abs=1e-4)

    def test_delay_broadcasts_into_a_gather(self):
        """One call lays a pulse on every trace at that trace's own arrival
        time — the shape a moveout gather needs."""
        time = np.linspace(0, 0.2, 2001)
        arrivals = np.array([0.05, 0.10, 0.15])

        gather = ricker_wavelet(time[:, None], 40.0, delay=arrivals[None, :])
        assert gather.shape == (time.size, arrivals.size)
        for column, expected in enumerate(arrivals):
            assert time[np.argmin(gather[:, column])] == pytest.approx(
                expected, abs=1e-4)

    def test_a_non_finite_delay_raises(self):
        with pytest.raises(ConfigurationError, match='delay must be finite'):
            ricker_wavelet(np.linspace(0, 0.1, 10), 40.0, delay=np.nan)


@pytest.mark.parametrize('module, name', [
    ('uacpy.acoustic_signal', n) for n in (
        'transfer_function_from_impulse_response', 'arrival_transfer_function',
        'broadband_propagation_loss', 'gate_transfer_function',
        'uniform_frequency_step', 'tone_phasor', 'simulate_arrival_reception',
        'rms_delay_spread', 'energy_support', 'coherence_bandwidth',
        'coherence_factor', 'channel_regime')] + [
    ('uacpy.comms', 'pulse_shaped_taps'),
    ('uacpy.acoustics', 'modal_attenuation')])
def test_the_delegation_label_lives_on_the_private_form(module, name):
    """``who`` names the entry point in a refusal when a method delegates.
    It is not physics, so the public signature does not carry it: the
    private ``_name`` takes it, keyword-only, and the public form refuses
    under its own name."""
    import importlib
    import inspect
    func = getattr(importlib.import_module(module), name)
    assert 'who' not in inspect.signature(func).parameters
    assert 'who' not in (func.__doc__ or '').split('Returns')[0].split()
    private = getattr(importlib.import_module(func.__module__), '_' + name)
    param = inspect.signature(private).parameters['who']
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def _signal_results():
    """One instance of every signal result tuple, with attributes set away
    from their defaults so a copy that dropped one would show."""
    from uacpy.acoustic_signal.beamforming import (
        BeamformedField, BeamformResult, Snapshots,
    )
    from uacpy.acoustic_signal.gathers import FKResult, RadonResult, TauPResult
    from uacpy.acoustic_signal.detect import AmbiguityResult
    from uacpy.acoustic_signal.timefreq import (
        Cepstrum, ComplexCepstrum, CWTResult, SpectrogramResult,
        WignerVilleResult,
    )
    from uacpy.comms.janus import JanusReception
    from uacpy.comms.link import BerCurve
    from uacpy.acoustic_signal.cqt import CQSpectrogramResult, CQTResult
    from uacpy.acoustic_signal.spectral import (
        ProbabilisticSpectralEstimate, SpectralEstimate,
    )
    from uacpy.noise.ambient import NoiseComponents
    x = np.arange(4.0)
    return {
        'SpectralEstimate': SpectralEstimate(
            x, x, scaling='exposure', method='constant_q',
            bands=[(1.0, 2.0, 3.0)], band_type='octave'),
        'ProbabilisticSpectralEstimate': ProbabilisticSpectralEstimate(
            x, x, x, mean_dB=x, level_step_dB=0.5, ref=1.0, scaling='spectrum',
            segment_levels_dB=np.ones((3, 4)), segment_times=x[:3]),
        'CQTResult': CQTResult(x, x + 1j),
        'CQSpectrogramResult': CQSpectrogramResult(x, x, x,
                                                   scaling='spectrum'),
        'WignerVilleResult': WignerVilleResult(x, x, x),
        'CWTResult': CWTResult(x, x, x),
        'Cepstrum': Cepstrum(x, x, sample_rate=2.0),
        'JanusReception': JanusReception(x, False, detected=True,
                                         start=3, doppler_scale=1e-4,
                                         statistic=x),
        'BerCurve': BerCurve(x, x, scheme='16qam', n_bits=10,
                             channel=[1.0, 0.5], n_train=7),
        'SpectrogramResult': SpectrogramResult(x, x, x, scaling='spectrum',
                                               mode='phase'),
        'ComplexCepstrum': ComplexCepstrum(x, 2),
        'AmbiguityResult': AmbiguityResult(x, x, x),
        'BeamformResult': BeamformResult(x, x, 1.0, ranges=x),
        'Snapshots': Snapshots(100.0, np.ones((2, 3))),
        'BeamformedField': BeamformedField(np.ones((2, 3)), x[:2], np.ones(3),
                                           None, grid_coords={'range': x[:3]}),
        'RadonResult': RadonResult(x, x, x, kind='hyperbolic'),
        'TauPResult': TauPResult(x, x, x),
        'FKResult': FKResult(x, x, x, x, scaling='power'),
        'NoiseComponents': NoiseComponents(x, x, x, x, x, x),
    }


_RESULT_NAMES = sorted(_signal_results())


class TestSignalResultsShareOneBehaviour:
    """Every signal result tuple copies, pickles and replaces the same way,
    keeping the attributes that say what it holds, and states the unit of
    each field."""

    @staticmethod
    def _attrs(result):
        return {name: getattr(result, name) for name in result._attrs}

    @pytest.mark.parametrize('name', _RESULT_NAMES)
    def test_a_pickle_round_trip_keeps_fields_and_attributes(self, name):
        import pickle
        r = _signal_results()[name]
        back = pickle.loads(pickle.dumps(r))
        assert type(back) is type(r)
        for mine, theirs in zip(back, r):
            np.testing.assert_array_equal(mine, theirs)
        assert repr(self._attrs(back)) == repr(self._attrs(r))

    @pytest.mark.parametrize('name', _RESULT_NAMES)
    def test_replace_changes_one_field_and_keeps_the_attributes(self, name):
        r = _signal_results()[name]
        field = r._fields[0]
        new = r._replace(**{field: 'changed'})
        assert type(new) is type(r)
        assert getattr(new, field) == 'changed'
        assert repr(self._attrs(new)) == repr(self._attrs(r))

    @pytest.mark.parametrize('name', _RESULT_NAMES)
    def test_replace_refuses_an_unknown_name(self, name):
        with pytest.raises(TypeError, match='unexpected field names'):
            _signal_results()[name]._replace(no_such_field=1)

    @pytest.mark.parametrize('name', _RESULT_NAMES)
    def test_the_units_name_every_field(self, name):
        r = _signal_results()[name]
        assert list(r.units) == list(r._fields)

    def test_the_units_follow_what_the_estimate_holds(self):
        r = _signal_results()
        assert r['SpectralEstimate'].units['power'] == 'Pa²·s'
        assert r['SpectralEstimate']._replace(
            scaling='density').units['power'] == 'Pa²/Hz'
        assert r['ProbabilisticSpectralEstimate'].units['level_edges'] == \
            'dB re 1 Pa²'
        assert r['ProbabilisticSpectralEstimate']._replace(
            ref=1e-6, scaling='density').units['level_edges'] == \
            'dB re 1 µPa²/Hz'
        assert r['SpectrogramResult'].units['power'] == 'rad'
        assert r['RadonResult'].units['moveout'] == 'm/s'
        assert r['FKResult'].units['power'] is None
        assert r['NoiseComponents'].units['wind'] == 'dB re 1 µPa²/Hz'

    def test_only_a_carrier_with_a_plotter_has_a_plot(self):
        r = _signal_results()
        assert not hasattr(r['BeamformResult'], 'plot')
        assert not hasattr(r['Snapshots'], 'plot')
        assert not hasattr(r['NoiseComponents'], 'plot')
        assert hasattr(r['FKResult'], 'plot')
