import warnings
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
    add_noise, fourier_synthesis, make_bandlimited_noise, make_noise_waveform,
)


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
        fs, fmin, fmax, T = 20_000.0, 500.0, 1500.0, 0.2
        t, s = lfm_chirp(fmin=fmin, fmax=fmax, duration=T, sample_rate=fs)
        assert len(t) == len(s)
        f_inst = self._instantaneous_frequency(s, 1.0 / fs)
        tm = 0.5 * (t[:-1] + t[1:])
        # Hilbert edge ripple contaminates the ends, so fit the (linear)
        # interior and read the endpoints off the fit: measured errors are
        # < 1e-4 Hz at both ends, so 1 Hz on a 1000 Hz sweep is generous.
        interior = (tm > 0.1 * T) & (tm < 0.9 * T)
        slope, intercept = np.polyfit(tm[interior], f_inst[interior], 1)
        assert intercept == pytest.approx(fmin, abs=1.0)
        assert slope * T + intercept == pytest.approx(fmax, abs=1.0)

    def test_hfm_chirp_frequency_is_hyperbolic_in_time(self):
        # HFM == linear *period* modulation: 1/f_inst(t) is linear in t,
        # running from 1/fmin to 1/fmax (Abraham §8.3.6's pulse).
        fs, fmin, fmax, T = 20_000.0, 500.0, 1500.0, 0.2
        t, s = hfm_chirp(fmin=fmin, fmax=fmax, duration=T, sample_rate=fs)
        period = 1.0 / self._instantaneous_frequency(s, 1.0 / fs)
        tm = 0.5 * (t[:-1] + t[1:])
        interior = (tm > 0.1 * T) & (tm < 0.9 * T)
        slope, intercept = np.polyfit(tm[interior], period[interior], 1)
        # Endpoint errors measured at 7e-8 s (P(0)) and 3e-8 s (P(T));
        # 1e-5 s against a 1.3e-3 s period span is generous.
        assert intercept == pytest.approx(1.0 / fmin, abs=1e-5)
        assert slope * T + intercept == pytest.approx(1.0 / fmax, abs=1e-5)
        # Discriminating half: the period really is linear (residual < 2 %
        # of the span; an LFM period fitted the same way leaves ~20 %).
        resid = period[interior] - (slope * tm[interior] + intercept)
        assert np.max(np.abs(resid)) < 0.02 * (1.0 / fmin - 1.0 / fmax)

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
                 'psd', 'ppsd', 'sel', 'spectrogram',
                 'SpectrogramResult', 'CWTResult', 'WignerVilleResult',
                 'FKResult', 'TauPResult', 'RadonResult'):
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


class TestDecidecadePartialBandsAreNaN:
    """A band the supplied grid does not fully cover was returned as the
    integral over the *covered part*, which is not that band's level —
    measured 3.8 dB (first band) and 3.2 dB (last) off their own trend on a
    flat PSD, and 5.5 dB low on the realistic ``psd() -> band_levels`` path.
    The one warning the function emitted counted a different condition
    (bands with <2 interior grid points), so it fired for bands that were
    fine and stayed silent for the two that were wrong."""

    @staticmethod
    def _flat(fmin=1.0, fmax=25000.0, df=0.25):
        f = np.arange(fmin, fmax, df)
        return f, np.ones_like(f)

    def test_fully_covered_bands_are_exact_and_partial_ones_are_nan(self):
        from uacpy.acoustic_signal.bands import (decidecade_band_levels,
                                                 decidecade_bands)
        f, psd_flat = self._flat()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, levels = decidecade_band_levels(psd_flat, f)
        lo, _, hi = decidecade_bands(f.min(), f.max())
        covered = (lo >= f.min()) & (hi <= f.max())
        exact = 10.0 * np.log10((hi - lo) / 1e-6 ** 2)
        # The discriminating half: the covered bands must stay exact, so the
        # fix cannot have been "widen a tolerance".
        assert np.nanmax(np.abs(levels[covered] - exact[covered])) < 1e-9
        assert np.all(np.isnan(levels[~covered]))

    def test_partial_bands_warn_and_name_the_support(self):
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f, psd_flat = self._flat()
        with pytest.warns(UserWarning, match='extend past the supplied'):
            decidecade_band_levels(psd_flat, f)

    def test_arrays_stay_parallel_with_decidecade_bands(self):
        # Shape contract: callers index the levels against a separately
        # computed decidecade_bands() with one mask, so dropping unsupported
        # bands would break them. nan keeps the arrays the same length.
        from uacpy.acoustic_signal.bands import (decidecade_band_levels,
                                                 decidecade_bands)
        f, psd_flat = self._flat()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centres, levels = decidecade_band_levels(psd_flat, f)
        lo, ctr, hi = decidecade_bands(f.min(), f.max())
        assert centres.shape == levels.shape == ctr.shape


class TestWaveformDegenerateInputs:
    """Every generator raises a typed ConfigurationError on degenerate
    parameters and accepts plain lists for time vectors."""

    def test_hfm_chirp_degenerate_parameters_raise(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):   # equal bounds divide by zero
            hfm_chirp(1000.0, 1000.0, 0.1, 8000.0)
        with pytest.raises(ConfigurationError):   # zero lower bound
            hfm_chirp(0.0, 500.0, 0.1, 8000.0)
        with pytest.raises(ConfigurationError):   # zero upper bound
            hfm_chirp(500.0, 0.0, 0.1, 8000.0)
        with pytest.raises(ConfigurationError):   # non-positive duration
            hfm_chirp(100.0, 500.0, 0.0, 8000.0)

    def test_hfm_chirp_down_sweep_still_allowed(self):
        t, s = hfm_chirp(2000.0, 100.0, 0.1, 8000.0)
        assert len(t) == len(s) > 0 and np.all(np.isfinite(s))

    def test_chirps_raise_instead_of_returning_empty(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            lfm_chirp(100.0, 500.0, -1.0, 8000.0)
        with pytest.raises(ConfigurationError):   # under one sample long
            lfm_chirp(100.0, 500.0, 1e-6, 8000.0)
        with pytest.raises(ConfigurationError):
            tone_burst(100.0, 0, 8000.0)

    def test_lfm_equal_bounds_is_a_pure_tone(self):
        t, s = lfm_chirp(500.0, 500.0, 0.1, 8000.0)
        np.testing.assert_allclose(s, np.sin(2 * np.pi * 500.0 * t))

    def test_pulses_reject_degenerate_parameters(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.waveforms import nwave, sparc_pulse
        time = np.linspace(0.0, 0.1, 64)
        with pytest.raises(ConfigurationError):
            nwave(time, 0.0)
        with pytest.raises(ConfigurationError):
            gaussian_pulse(time, 0.05, 0.0)
        with pytest.raises(ConfigurationError):
            ricker_wavelet(time, 0.0)
        with pytest.raises(ConfigurationError):
            sparc_pulse(time, 0.0, "R")

    def test_time_vector_functions_accept_lists(self):
        from uacpy.acoustic_signal.waveforms import nwave, sparc_pulse
        tl = [0.0, 0.001, 0.002, 0.005]
        assert ricker_wavelet(tl, 100.0).shape == (4,)
        assert gaussian_pulse(tl, 0.002, 0.001).shape == (4,)
        assert nwave(tl, 100.0).shape == (4,)
        assert sparc_pulse(tl, 2 * np.pi * 100.0, "R")[0].shape == (4,)


def test_synthesize_noise_returns_the_rate_the_time_axis_uses():
    """The returned sample rate is the float rate the time axis was built
    from, also when the default 2*Fxx[-1] is not an integer."""
    from uacpy.acoustic_signal.noise_synthesis import synthesize_noise_from_psd
    f = np.array([1.0, 10.3])
    t, x, fs = synthesize_noise_from_psd(
        np.array([1e-6, 1e-6]), f, duration=0.5,
        rng=np.random.default_rng(0))
    assert isinstance(fs, float) and fs == 2 * 10.3
    assert abs(1.0 / (t[1] - t[0]) - fs) < 1e-9


class TestFRFReservedKwargs:
    """Welch options that the FRF sets internally are rejected typed, not
    left to die in scipy as a bare TypeError."""

    def test_scaling_and_fs_raise_configurationerror(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.system_id import FRF
        with pytest.raises(ConfigurationError, match="scaling"):
            FRF(method="welch", scaling="spectrum")
        with pytest.raises(ConfigurationError, match="fs"):
            FRF(method="welch", fs=48_000.0)

    def test_legitimate_welch_kwargs_still_pass_through(self):
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(2)
        x = rng.standard_normal(8192)
        y = np.convolve(x, [1.0, 0.5], mode="same")
        freqs, tf = FRF(method="welch", nperseg=1024,
                        window="hamming").compute(x, y, 8000.0)
        assert freqs.size == 513 and np.all(np.isfinite(tf))

    def test_etfe_grid_is_the_full_record_grid(self):
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(3)
        x = rng.standard_normal(16384)
        y = np.convolve(x, [1.0, 0.5], mode="same")
        f_etfe, _ = FRF(method="etfe").compute(x, y, 8000.0)
        f_welch, _ = FRF(method="welch").compute(x, y, 8000.0)
        np.testing.assert_allclose(
            f_etfe, np.fft.rfftfreq(x.size, d=1 / 8000.0))
        assert f_welch.size == 8192 // 2 + 1      # the nperseg grid


def test_mseq_polarity_matches_dsss_m_sequence():
    """Both m-sequence generators use the standard BPSK mapping
    s = 1 - 2*bit (bit 0 -> +1, bit 1 -> -1): a full period sums to -1
    (2**(m-1) ones map to -1), and despreading with either family's code
    keeps the symbol sign."""
    from uacpy.acoustic_signal.sequences import mseq
    from uacpy.comms.dsss import m_sequence, spread, despread
    s = mseq(5)
    d = m_sequence(5, [5, 2])
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
    from uacpy.acoustic_signal.sequences import bpsk_modulate
    from uacpy.acoustic_signal.noise_synthesis import synthesize_noise_from_psd
    from uacpy.acoustic_signal.system_id import FRF
    with pytest.raises(ConfigurationError):       # chip rate of zero
        bpsk_modulate(np.array([1, -1]), 100.0, 1000.0, 0.0)
    with pytest.raises(ConfigurationError):       # zero-length realisation
        synthesize_noise_from_psd(np.ones(8), np.linspace(10, 100, 8),
                                  duration=0)
    with pytest.raises(ConfigurationError):       # burst rounds to 0 samples
        tone_burst(1000.0, 1, 400.0)
    rng = np.random.default_rng(5)
    u = rng.standard_normal(64)
    y = np.convolve(u, [1.0, 0.5], mode="full")[:64]
    with pytest.raises(ConfigurationError, match="m .*must be <= N"):
        FRF().compute_lsfir(y, u, 1000.0, m=100, N=64)


def test_add_noise_accepts_a_plain_list():
    out = add_noise([0.0] * 1000, 1000.0, 100.0, 50.0, 100.0, 50.0,
                    rng=np.random.default_rng(6))
    assert isinstance(out, np.ndarray) and out.shape == (1000,)
    assert np.all(np.isfinite(out)) and np.std(out) > 0


class TestSparcPulseLibraryShapes:
    """The 11-letter SPARC pulse library accepts every documented code and
    gates each pulse as ``cans.m`` does."""

    @pytest.mark.parametrize('code', list('PRASHNMGTCE'))
    def test_all_eleven_shapes_accepted(self, code):
        from uacpy.acoustic_signal.waveforms import sparc_pulse
        t = np.linspace(-0.05, 0.1, 512)
        s, title = sparc_pulse(t, 2 * np.pi * 100.0, code)
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
        from uacpy.acoustic_signal.waveforms import sparc_pulse
        with pytest.raises(ConfigurationError, match='Unknown pulse type'):
            sparc_pulse(np.linspace(0, 0.1, 64), 2 * np.pi * 100.0, 'Z')

    def test_nwave_is_gated_to_one_period(self):
        from uacpy.acoustic_signal.waveforms import nwave
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


class TestMseqBounds:
    """``mseq`` order bounds and chip alphabet."""

    def test_mseq_rejects_out_of_range_order(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.sequences import mseq
        for bad in (0, 1, 16, -3):
            with pytest.raises(ConfigurationError, match='between 2 and 15'):
                mseq(bad)

    def test_mseq_chips_are_plus_minus_one_with_full_length(self):
        from uacpy.acoustic_signal.sequences import mseq
        for m in (2, 7, 15):
            s = mseq(m)
            assert len(s) == 2 ** m - 1
            assert set(np.unique(s)) == {-1.0, 1.0}


class TestMakeMseqProbe:
    """Structure of the channel-sounding probe (docs/guide/signal.md §8):
    0.2 s zero leader, whole BPSK'd ``mseq(10)`` periods at chip rate
    ``(fmax − fmin)/2`` on carrier ``(fmin + fmax)/2``, normalised to 0.95
    full scale and zero-filled to exactly ``round(T_tot·fs)`` samples."""

    FS = 10_000.0
    FMIN, FMAX = 1000.0, 2000.0    # fc = 1500 Hz, 500 chips/s → 20 smp/chip
    N_PERIOD = 1023 * 20           # mseq(10) → 1023 chips

    def _probe(self, T_tot=5.0):
        from uacpy.acoustic_signal.sequences import make_mseq_probe
        return make_mseq_probe(self.FMIN, self.FMAX, self.FS, T_tot)

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
        # Chip rate (fmax − fmin)/2 puts the BPSK main lobe (first sinc
        # nulls) exactly on [fmin, fmax]; measured in-band fraction 0.91.
        assert power[in_band].sum() / power.sum() > 0.85
        assert self.FMIN <= freqs[int(np.argmax(power))] <= self.FMAX

    def test_too_short_for_leader_plus_one_period_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        # One period lasts 2.046 s, so 1 s cannot hold leader + period.
        with pytest.raises(ConfigurationError, match='too short'):
            self._probe(T_tot=1.0)


class TestFourierSynthesis:
    """AT ``stack.m`` translation: raw-DFT synthesis on the input frequency
    grid with the one-sided spectrum doubled, plus the ``Tstart`` phase-ramp
    warning (docs/guide/signal.md §9)."""

    def test_single_bin_synthesises_the_expected_cosine(self):
        N, df = 64, 10.0
        freqs = np.arange(N) * df           # grid starts at DC: no warning
        H = np.zeros(N, dtype=complex)
        H[8] = 3.0                          # one 80 Hz bin, real amplitude
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            t, x = fourier_synthesis(H, freqs)
        assert t[0] == 0.0 and t.size == x.size == N
        # 2·Re{ifft}: a single one-sided bin of amplitude A comes back as
        # (2A/N)·cos(2π f t) on the DFT time grid, exact to machine eps.
        np.testing.assert_allclose(
            x, (2.0 * 3.0 / N) * np.cos(2 * np.pi * 80.0 * t), atol=1e-12)

    def test_offset_band_without_tstart_warns_of_phase_ramp(self):
        freqs = np.arange(10.0, 100.0, 10.0)     # frequencies[0] > 0
        H = np.ones(freqs.size, dtype=complex)
        with pytest.warns(UserWarning, match='phase ramp'):
            fourier_synthesis(H, freqs)

    def test_tstart_silences_the_warning_and_anchors_the_time_axis(self):
        freqs = np.arange(10.0, 100.0, 10.0)
        H = np.ones(freqs.size, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            t, x = fourier_synthesis(H, freqs, Tstart=0.5)
        assert t[0] == pytest.approx(0.5)
        assert x.shape == freqs.shape
