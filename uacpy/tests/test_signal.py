"""
Smoke tests for uacpy.acoustic_signal (uacpy.acoustic_signal).

Bare minimum that the public API is reachable and behaves on simple inputs.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal.waveforms import (
    gaussian_pulse, hfm_chirp, lfm_chirp, ricker_wavelet, tone_burst,
)
from uacpy.acoustic_signal.noise_synthesis import (
    add_noise, make_bandlimited_noise, make_noise_waveform,
)


class TestGenerators:
    """Pulse / chirp generators return finite samples on plausible inputs."""

    def test_gaussian_pulse_shape(self):
        time = np.linspace(0, 0.1, 1024)
        s = gaussian_pulse(time, delay=0.05, duration=0.01)
        assert len(s) == len(time)
        assert np.all(np.isfinite(s))

    def test_lfm_chirp_runs(self):
        fs = 10_000.0
        t, s = lfm_chirp(fmin=100, fmax=2000, duration=0.1, sample_rate=fs)
        assert len(t) == len(s)
        assert np.all(np.isfinite(s))

    def test_hfm_chirp_runs(self):
        fs = 10_000.0
        t, s = hfm_chirp(fmin=100, fmax=2000, duration=0.1, sample_rate=fs)
        assert len(t) == len(s)
        assert np.all(np.isfinite(s))

    def test_ricker_wavelet_runs(self):
        time = np.linspace(0, 0.1, 1024)
        s = ricker_wavelet(time, frequency=200.0)
        assert len(s) == len(time)
        assert np.all(np.isfinite(s))

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

    def test_add_noise_adds_band_limited_noise(self):
        """The added term is noise inside the requested band.

        A silent input makes the output the noise term alone, so the band
        occupancy and the sample-to-sample correlation both describe the noise
        directly. Variance alone cannot tell noise from any other waveform.
        """
        fs, fc, bw = 48_000.0, 10_000.0, 10_000.0
        y = np.asarray(add_noise(
            np.zeros(8192), sample_rate=fs, source_level=0.0,
            noise_level=40.0, fc=fc, bandwidth=bw,
        )).ravel()

        freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
        power = np.abs(np.fft.rfft(y)) ** 2
        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        assert power[in_band].sum() / power.sum() > 0.9
        assert freqs[np.argmax(power)] > fc - bw / 2

        # Any monotone or otherwise smooth waveform correlates ~1 at lag 1.
        assert abs(np.corrcoef(y[:-1], y[1:])[0, 1]) < 0.7

    def test_add_noise_realisations_are_seeded_and_per_receiver(self):
        """``rng`` selects the realisation, and receivers are independent.

        The docstring's array-gain contract needs zero cross-channel
        correlation, so a shared realisation across columns is a defect.
        """
        fs = 48_000.0
        kw = dict(source_level=0.0, noise_level=40.0, fc=10_000.0,
                  bandwidth=10_000.0)
        x = np.zeros(4096)
        first = np.asarray(add_noise(x, sample_rate=fs, **kw,
                                     rng=np.random.default_rng(3)))
        again = np.asarray(add_noise(x, sample_rate=fs, **kw,
                                     rng=np.random.default_rng(3)))
        other = np.asarray(add_noise(x, sample_rate=fs, **kw,
                                     rng=np.random.default_rng(4)))
        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)

        block = np.asarray(add_noise(np.zeros((4096, 4)), sample_rate=fs, **kw))
        corr = np.corrcoef(block.T)
        assert np.max(np.abs(corr[~np.eye(4, dtype=bool)])) < 0.2

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

    def test_make_noise_waveform_is_1d_with_consistent_length(self):
        fs, T = 10_000.0, 0.1
        t, n = make_noise_waveform(fc=1000.0, bandwidth_hz=500.0, T=T, sample_rate=fs)
        # Returns (time, signal) like the tonal generators (tone_burst, …);
        # 1-D, length int(T*fs), with the time axis the same length (no
        # carrier/noise mismatch from arange-vs-int float drift).
        assert n.ndim == 1
        assert n.shape == t.shape == (int(T * fs),)
        assert np.all(np.isfinite(n))
        assert t[0] == 0.0 and np.allclose(np.diff(t), 1.0 / fs)

    def test_make_noise_waveform_unresolvable_duration_raises(self):
        """``T*bandwidth_hz < 1`` gives a zero-length white-noise draw, which
        scipy.signal.resample cannot consume. Reject it up front."""
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            make_noise_waveform(fc=1000.0, bandwidth_hz=500.0, T=1e-4,
                                sample_rate=10_000.0)

    def test_noise_generators_are_reproducible_from_a_seeded_rng(self):
        """Every noise generator takes an ``rng=``, like the ``uacpy.comms``
        side, so a realisation can be reproduced independently of global
        numpy state."""
        from uacpy.acoustic_signal.noise_synthesis import (
            synthesize_noise_from_psd)
        fs, dur = 10_000.0, 0.1
        kw = dict(fc=1000.0, bandwidth=500.0, duration=dur, sample_rate=fs)
        a = make_bandlimited_noise(**kw, rng=np.random.default_rng(7))[1]
        b = make_bandlimited_noise(**kw, rng=np.random.default_rng(7))[1]
        c = make_bandlimited_noise(**kw, rng=np.random.default_rng(8))[1]
        assert np.array_equal(a, b) and not np.array_equal(a, c)

        wkw = dict(fc=1000.0, bandwidth_hz=500.0, T=dur, sample_rate=fs)
        assert np.array_equal(
            make_noise_waveform(**wkw, rng=np.random.default_rng(7))[1],
            make_noise_waveform(**wkw, rng=np.random.default_rng(7))[1])

        nkw = dict(sample_rate=fs, source_level=120.0, noise_level=80.0,
                   fc=1000.0, bandwidth=200.0)
        assert np.array_equal(
            add_noise(np.zeros(1024), **nkw, rng=np.random.default_rng(7)),
            add_noise(np.zeros(1024), **nkw, rng=np.random.default_rng(7)))

        f = np.logspace(1, 3, 32)
        pkw = dict(duration=0.05, n_fft=1024, sample_rate=fs)
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
        from uacpy.acoustic_signal.bands import (decidecade_bands,
                                                 decidecade_band_levels)
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER as REF
        f = np.linspace(1, 20000, 40000)
        psd = np.full_like(f, 1e-12)
        c, lv = decidecade_band_levels(psd, f)
        lo, _, hi = decidecade_bands(f.min(), f.max())
        exact = 10 * np.log10(1e-12 * (hi - lo) / REF ** 2)
        covered = (lo >= f.min()) & (hi <= f.max()) & np.isfinite(lv)
        assert covered.sum() > 30
        np.testing.assert_allclose(lv[covered], exact[covered], atol=0.01)

    def test_white_noise_band_levels_rise_1db_per_band(self):
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.linspace(1, 20000, 40000)
        psd = np.ones_like(f) * 1e-12               # flat Pa^2/Hz
        c, lv = decidecade_band_levels(psd, f)
        step = np.diff(lv[(c > 200) & (c < 5000)])
        assert np.allclose(step, 1.0, atol=0.05)    # each band 10^0.1 wider -> +1 dB

    def test_bands_validate_input(self):
        from uacpy.acoustic_signal.bands import decidecade_bands
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            decidecade_bands(1000, 100)

    def test_coarse_grid_falls_back_not_nan(self):
        # A coarse, log-spaced grid leaves low bands with one sample; instead of
        # a silent NaN they get a rectangular estimate and a single warning.
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.logspace(np.log10(10), np.log10(2000), 25)
        psd = np.ones_like(f) * 1e-12
        with pytest.warns(UserWarning, match="too coarse"):
            c, lv = decidecade_band_levels(psd, f)
        # coarse-grid bands still resolve a finite level
        assert np.any(np.isfinite(lv[c < 100]))


def test_acoustic_signal_is_importable():
    import uacpy.acoustic_signal as sig
    import uacpy
    assert sig is uacpy.acoustic_signal


def test_signal_symbols_resolve():
    import uacpy
    for name in ('lfm_chirp', 'hfm_chirp', 'tone_burst', 'gaussian_pulse',
                 'ricker_wavelet', 'add_noise', 'make_bandlimited_noise',
                 'psd', 'ppsd', 'sel', 'spectrogram'):
        assert hasattr(uacpy.acoustic_signal, name), \
            f"uacpy.acoustic_signal.{name} missing"


class TestSEL:
    """SEL must integrate power (Parseval-exact), not over-read the way a
    coherent-normalization Hann taper would (+1.76 dB on stationary signals,
    and an impulse at a segment boundary annihilated)."""

    def test_tone_exposure_is_parseval_exact(self):
        from uacpy.acoustic_signal.analysis import sel as sel_fn
        fs = 48000
        t = np.arange(fs) / fs
        x = 2.0 * np.sin(2 * np.pi * 1000.0 * t)   # exposure = A^2/2 * T = 2.0
        sel, _ = sel_fn(x, fs, band_type='third_octave', fmin=10, fmax=20000,
                        nfft=fs)
        assert sel.sum() == pytest.approx(np.sum(x ** 2) / fs, rel=1e-6)

    def test_impulse_not_annihilated(self):
        from uacpy.acoustic_signal.analysis import sel as sel_fn
        fs = 48000
        imp = np.zeros(fs)
        imp[0] = 10.0   # a Hann-windowed single segment would zero this out
        sel, _ = sel_fn(imp, fs, band_type='linear', fmin=1.0, fmax=fs / 2,
                        num_bands=240, nfft=fs)
        # full-band exposure ≈ Σx²/fs (only the excluded DC bin is dropped)
        assert sel.sum() == pytest.approx(np.sum(imp ** 2) / fs, rel=1e-3)

    def test_coarse_bands_do_not_double_count_bins(self):
        # 1-Hz FFT bins (nfft=fs) against sub-bin-wide low third-octave bands:
        # each bin must contribute to exactly one band, so a flat tone's total
        # exposure is conserved (no bin double-counted across overlapping bands).
        from uacpy.acoustic_signal.analysis import sel as sel_fn
        fs = 1000
        t = np.arange(fs) / fs
        x = np.sin(2 * np.pi * 50.0 * t)
        sel, _ = sel_fn(x, fs, band_type='third_octave', fmin=8.9125, fmax=400,
                        nfft=fs)
        assert sel.sum() == pytest.approx(np.sum(x ** 2) / fs, rel=1e-6)


class TestFRF:
    """FRF automatic FIR-order selection (m='AIC'|'BIC'|'FPE'|'CP') must run,
    not crash with 'count >= None' from an un-defaulted stop_count."""

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_auto_order_runs_and_recovers_order(self, criterion):
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(1)
        u = rng.standard_normal(2000)
        g = np.array([1.0, -0.5, 0.25])                  # order-3 FIR
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(2000)
        frf = FRF()
        _, tf = frf.compute(u, y, 1000.0, method='ls_fir', m=criterion)
        assert np.isfinite(tf).all()
        # every criterion recovers the true order-3 FIR at this SNR; the
        # criterion itself stays on .m, the chosen order is published
        # separately so a reused FRF re-selects instead of pinning.
        assert frf.selected_order == 3
        assert frf.m == criterion

    def test_cp_recovers_order_six_fir(self):
        """Mallows' Cp recovers the true order-6 FIR at moderate SNR. Cp scales
        the residual sum of squares by σ̂², the residual variance of a low-bias
        reference fit; this higher-order case exercises that estimate (order 3
        at high SNR above is too easy to constrain it).
        """
        from uacpy.acoustic_signal.system_id import FRF
        r = np.random.default_rng(2)
        N, order = 3000, 6
        u = r.standard_normal(N)
        g = r.standard_normal(order)
        g = g / np.linalg.norm(g)
        clean = np.convolve(u, g)[:N]
        y = clean + 0.1 * np.std(clean) * r.standard_normal(N)
        frf = FRF()
        _, tf = frf.compute(u, y, 1000.0, method='ls_fir', m='CP')
        assert np.isfinite(tf).all()
        assert frf.selected_order == order
        assert frf.m == 'CP'

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_order_selection_is_amplitude_scale_invariant(self, criterion):
        """The selected order must depend on the data, not on its units: a
        pressure record in Pa and the same record in MPa must give the same
        FIR order. All four criteria compare log(sse) or sse ratios, so only
        the exact-fit cutoff can break the invariance."""
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(0)
        u = rng.standard_normal(400)
        g = np.array([1.0, 0.5, -0.3])
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(400)
        orders = []
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            frf = FRF()
            frf.compute(scale * u, scale * y, 1000.0, method='ls_fir',
                        m=criterion, m_max=60)
            orders.append(frf.selected_order)
        assert orders == [3, 3, 3, 3]

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_exact_fit_selects_lowest_explaining_order(self, criterion):
        """A pure-gain loopback y = 2*u is fitted exactly at order 1; the
        search must return that order instead of discarding every candidate."""
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(3)
        u = rng.standard_normal(400)
        frf = FRF()
        _, tf = frf.compute(u, 2.0 * u, 1000.0, method='ls_fir', m=criterion,
                            m_max=60)
        assert np.isfinite(tf).all()
        assert frf.selected_order == 1
        assert np.asarray(frf.g) == pytest.approx([2.0])

    def test_unfittable_input_raises_configurationerror(self):
        """An all-zero input is singular at every order: typed error, not a
        ValueError out of scipy.signal.freqz on a None filter."""
        from uacpy.acoustic_signal.system_id import FRF
        from uacpy.core.exceptions import ConfigurationError
        rng = np.random.default_rng(4)
        y = rng.standard_normal(300)
        with pytest.raises(ConfigurationError):
            FRF().compute(np.zeros(300), y, 1000.0, method='ls_fir', m='AIC',
                          m_max=40)

    def test_zero_row_input_raises_configurationerror(self):
        """A 2-D input with no measurement rows must not fall through the
        per-measurement loop and hit an UnboundLocalError on the frequency
        axis."""
        from uacpy.acoustic_signal.system_id import FRF
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            FRF().compute(np.zeros((0, 100)), np.zeros((0, 100)), 1000.0,
                          method='ls_fir', m=4)

    def test_method_switch_clears_ls_fir_state(self):
        """``selected_order``/``g`` are ls_fir-only and ``coh`` is welch-only;
        a reused FRF must not report the previous method's values."""
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        frf = FRF()
        frf.compute(u, y, 1000.0, method='ls_fir', m='AIC', m_max=40)
        assert frf.selected_order is not None
        assert frf.coh is None
        frf.compute(u, y, 1000.0, method='welch', nperseg=512)
        assert frf.selected_order is None and frf.g == 0
        assert frf.coh is not None and frf.coh.shape == frf.frequencies.shape


def test_degenerate_input_guards_raise_configurationerror():
    """Pre-production robustness: degenerate inputs raise a typed
    ConfigurationError, not a raw ValueError/ZeroDivisionError."""
    from uacpy.core.exceptions import ConfigurationError
    from uacpy.acoustic_signal import sel, cwt, tone_burst
    with pytest.raises(ConfigurationError):       # sel: empty data
        sel(np.array([]), 48000.0)
    with pytest.raises(ConfigurationError):       # sel: zero integration_time
        sel(np.ones(2000), 48000.0, integration_time=0.0)
    with pytest.raises(ConfigurationError):       # cwt: signal too short (n<8)
        cwt(np.ones(5), 8000.0)
    with pytest.raises(ConfigurationError):       # tone_burst: frequency 0
        tone_burst(0.0, 5, 1000.0)


class TestFRFEstimators:
    """H1 and H2 differ only in which noise they reject (Bendat & Piersol):
    ``H1 = Sxy/Sxx`` is unbiased when the noise is on the output, ``H2 =
    Syy/Syx`` when it is on the input. Both recover the plant when there is no
    noise at all."""

    FS, N = 8000.0, 200_000
    H = np.array([1.0, -0.7, 0.35, -0.1])

    @classmethod
    def _truth(cls, f):
        n = np.arange(cls.H.size)
        return (cls.H[None, :] * np.exp(-2j * np.pi * np.outer(f, n) / cls.FS)
                ).sum(axis=1)

    @classmethod
    def _err(cls, estimator, x, y):
        from uacpy.acoustic_signal.system_id import FRF
        f, tf, coh = FRF(estimator=estimator,
                         nperseg=4096).compute_welch(x, y, cls.FS)
        band = (f > 100) & (f < 3500)
        return (float(np.abs(tf[band] - cls._truth(f)[band]).max()),
                float(coh[band].mean()))

    def _signals(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.N)
        return rng, x, np.convolve(x, self.H)[: self.N]

    def test_both_recover_the_plant_without_noise(self):
        # Noise-free, so the residual is Welch segmentation/leakage only:
        # measured max |tf - truth| is 4.6e-4 and the coherence is 1 - 4e-7.
        # 1e-2 / 1e-3 are floors an order of magnitude above that.
        _, x, y = self._signals(5)
        for est in ('H1', 'H2'):
            err, coh = self._err(est, x, y)
            assert err < 1e-2 and coh == pytest.approx(1.0, abs=1e-3)

    def test_h1_beats_h2_on_output_noise(self):
        # H2 is biased UP by output noise: measured errors are H1 0.21 vs
        # H2 0.91, a factor 4.3, so the required factor 3 leaves ~40 % margin
        # on this seed.
        rng, x, y = self._signals(6)
        yn = y + 0.5 * rng.standard_normal(self.N)
        assert self._err('H1', x, yn)[0] < self._err('H2', x, yn)[0] / 3

    def test_h2_beats_h1_on_input_noise(self):
        # The mirror case is weaker: measured H1 0.61 vs H2 0.39, a factor
        # 1.57 against the required 1.5 — only ~5 % margin, so this assertion
        # is seed-sensitive and the factor cannot be tightened.
        rng, x, y = self._signals(7)
        xn = x + 0.5 * rng.standard_normal(self.N)
        assert self._err('H2', xn, y)[0] < self._err('H1', xn, y)[0] / 1.5


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
        _t, n = make_bandlimited_noise(fc, bw, 2.0, fs,
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

    def test_add_noise_realises_the_requested_level_in_band(self):
        """Pre-fix this read 5.2 dB against a requested 40 dB, because the
        level was scaled from the same relocated design and so was internally
        self-consistent — which is why nothing downstream noticed."""
        fs, fc, bw, level = self.FS, 100.0, 100.0, 40.0
        y = np.asarray(add_noise(np.zeros(int(2 * fs)), fs, 0.0, level, fc, bw,
                                 rng=np.random.default_rng(0))).ravel()
        freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
        psd = np.abs(np.fft.rfft(y)) ** 2 * 2.0 / (fs * y.size)
        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        assert 10 * np.log10(psd[in_band].mean()) == pytest.approx(level, abs=3.0)

    @pytest.mark.parametrize('fc,bw,fs', [(10.0, 100.0, 48_000.0),
                                          (23_990.0, 100.0, 48_000.0),
                                          (100.0, 100.0, 150.0)])
    def test_an_unrealisable_band_is_refused_not_moved(self, fc, bw, fs):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='not realisable'):
            make_bandlimited_noise(fc, bw, 0.5, fs)
