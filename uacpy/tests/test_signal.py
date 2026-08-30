"""Tests for ``uacpy.acoustic_signal``: waveform generators, noise synthesis,
and the ``FRF`` system-identification estimators.

Covers that the public API is reachable and behaves on simple inputs, that
each generator produces the waveform it documents rather than merely finite
samples, and that the entry points refuse the inputs they cannot represent —
a sweep bound below zero, a degenerate fit the criterion cannot report on.
"""

import warnings

import numpy as np
import pytest

from uacpy.acoustic_signal.system_id import FRF
from uacpy.acoustic_signal.waveforms import (
    gaussian_pulse, hfm_chirp, lfm_chirp, ricker_wavelet, tone_burst,
)
from uacpy.acoustic_signal.noise_synthesis import (
    add_noise, fourier_synthesis, make_bandlimited_noise, make_noise_waveform,
    synthesize_noise_from_psd,
)
from uacpy.comms.modulation import (
    fsk_demodulate as _fsk_demodulate,
    fsk_modulate as _fsk_modulate,
)
from uacpy.core.exceptions import ConfigurationError

NAN = float('nan')
#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]
from uacpy.acoustic_signal.active import (ambiguity_function,
                                          pulse_compression)
from uacpy.acoustic_signal.analysis import ppsd, psd, sel
from uacpy.acoustic_signal.sequences import bpsk_modulate
from uacpy.acoustic_signal.timefreq import (cwt, instantaneous_frequency,
                                            spectrogram)
from uacpy.acoustic_signal.transforms import (
    fk_transform,
    radon_transform as _radon_transform,
    taup_transform as _taup_transform,
)
from uacpy.acoustic_signal.timefreq import (
    cepstrum as _cepstrum,
    envelope as _envelope,
    wigner_ville as _wigner_ville,
)
from uacpy.acoustic_signal.constant_q import (
    constant_q_psd as _constant_q_psd,
)
from uacpy.acoustic_signal.modal import (
    warp_signal as _warp_signal,
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
        fs, dur = 10_000.0, 0.1
        t, n = make_noise_waveform(
            fc=1000.0, bandwidth=500.0, duration=dur, sample_rate=fs)
        # Returns (time, signal) like the tonal generators (tone_burst, …);
        # 1-D, length int(duration*fs), with the time axis the same length
        # (no carrier/noise mismatch from arange-vs-int float drift).
        assert n.ndim == 1
        assert n.shape == t.shape == (int(dur * fs),)
        assert np.all(np.isfinite(n))
        assert t[0] == 0.0 and np.allclose(np.diff(t), 1.0 / fs)

    def test_both_band_noise_generators_take_one_shared_kwargs_dict(self):
        """``make_noise_waveform`` and ``make_bandlimited_noise`` are siblings:
        same argument order, same return shape, same purpose. They spell the
        band and the length ``bandwidth`` and ``duration``, so one kwargs dict
        drives both and a caller can swap one for the other in place."""
        import inspect
        kw = dict(fc=1000.0, bandwidth=500.0,
                  duration=0.1, sample_rate=10_000.0)

        t_w, n_w = make_noise_waveform(**kw)
        t_b, n_b = make_bandlimited_noise(**kw)
        assert n_w.shape == n_b.shape == t_w.shape == t_b.shape == (1000,)

        assert (list(inspect.signature(make_noise_waveform).parameters)
                == list(inspect.signature(make_bandlimited_noise).parameters)
                == ['fc', 'bandwidth', 'duration', 'sample_rate', 'rng'])

    def test_make_noise_waveform_unresolvable_duration_raises(self):
        """``duration*bandwidth < 1`` gives a zero-length white-noise draw,
        which scipy.signal.resample cannot consume. Reject it up front."""
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            make_noise_waveform(fc=1000.0, bandwidth=500.0, duration=1e-4,
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

        assert np.array_equal(
            make_noise_waveform(**kw, rng=np.random.default_rng(7))[1],
            make_noise_waveform(**kw, rng=np.random.default_rng(7))[1])

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
        # chosen order is published on .selected_order so a reused FRF
        # re-selects instead of pinning, and the per-call criterion leaves
        # the object's own .m as the constructor set it.
        assert frf.selected_order == 3
        assert frf.m == FRF().m

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
        assert frf.m == FRF().m

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

    def test_the_structural_end_bands_are_nan_and_are_not_warned_about(self):
        """The band set keeps every band *overlapping* the support, so the
        first and last are partial on any grid whose ends do not land on
        decidecade band edges — which no rfftfreq grid does. Their ``nan``
        level is the diagnostic; a warning about them fires on every
        well-formed call and cannot distinguish a short grid from a call."""
        from uacpy.acoustic_signal.bands import (decidecade_band_levels,
                                                 decidecade_bands)
        f, psd_flat = self._flat()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, levels = decidecade_band_levels(psd_flat, f)
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
            decidecade_band_levels(np.ones_like(f) * 1e-12, f)

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

    def test_hfm_chirp_down_sweep_allowed(self):
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

    def test_the_generators_use_the_shared_signal_layer_scalar_guard(self):
        """Every waveform scalar goes through
        ``_signal_validate.require_positive_finite_scalar`` — the same guard
        the other ``acoustic_signal`` modules apply — so the message names the
        parameter's unit and a non-finite value is refused, not only a
        non-positive one. A private copy of the check inside ``waveforms``
        gives neither: ``inf`` passes ``value > 0`` and produces silent
        garbage (an all-zero ``nwave``, a NaN chirp)."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.waveforms import nwave, sparc_pulse
        time = np.linspace(0.0, 0.1, 64)
        for call, pattern in (
                (lambda: sparc_pulse(time, np.inf, "R"),
                 r"sparc_pulse: omega must be > 0 rad/s and finite"),
                (lambda: ricker_wavelet(time, np.inf),
                 r"ricker_wavelet: frequency must be > 0 Hz and finite"),
                (lambda: gaussian_pulse(time, 0.05, np.nan),
                 r"gaussian_pulse: duration must be > 0 s and finite"),
                (lambda: lfm_chirp(100.0, 500.0, np.inf, 8000.0),
                 r"lfm_chirp: duration must be > 0 s and finite"),
                (lambda: tone_burst(100.0, np.inf, 8000.0),
                 r"tone_burst: n_cycles must be > 0 cycles and finite"),
                (lambda: hfm_chirp(100.0, np.inf, 0.1, 8000.0),
                 r"hfm_chirp: fmax must be > 0 Hz and finite"),
                (lambda: nwave(time, np.inf),
                 r"nwave: frequency must be > 0 Hz and finite"),
        ):
            with pytest.raises(ConfigurationError, match=pattern):
                call()

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

    def test_legitimate_welch_kwargs_pass_through(self):
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
    with pytest.raises(ConfigurationError):       # tone at/above Nyquist
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


class TestLsFirCpReferenceFitReportsDegenerateInput:
    N = 256

    def _degenerate(self, criterion):
        return FRF(method='ls_fir').compute_lsfir(
            np.zeros(self.N), np.ones(self.N), 1000.0, criterion, self.N)

    def test_cp_raises_the_typed_error_the_other_criteria_reach(self):
        # The Cp reference fit runs before the candidate loop's LinAlgError
        # handling, so a constant input escaped as a raw LinAlgError.
        with pytest.raises(ConfigurationError, match="criterion 'CP'"):
            self._degenerate('CP')

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE'])
    def test_the_other_criteria_return_a_non_empty_result_on_the_same_input(self, criterion):
        assert self._degenerate(criterion)[0].size > 0


class TestLsFirSolvesSingularNormalEquationsByMinimumNorm:
    """``compute_lsfir`` fits through ``X.T @ X``, whose condition number is
    ``cond(X)**2``, so a probe that leaves part of the Nyquist band unexcited
    makes the system numerically singular at any FIR order longer than the
    excited band supports. The shipped default ``m=512`` reaches it on an
    ordinary 100 Hz - 20 kHz sweep at fs = 48 kHz (``cond(X) = 4.2e11``,
    reciprocal condition number of the information matrix 1e-20): the LU
    solve of that system carries no correct digit, and the coefficients come
    back from a rank-revealing least-squares solve of the same equations
    instead, with the order named in a warning.
    """

    fs = 48000.0
    N = 8000
    m = 512
    h_true = np.array([1.0, -0.6, 0.3, 0.1])

    def _fit(self):
        import scipy.signal as sig
        rng = np.random.default_rng(11)
        u = sig.chirp(np.arange(self.N) / self.fs, 100.0,
                      self.N / self.fs, 20000.0)
        y = (np.convolve(u, self.h_true)[:self.N]
             + 1e-8 * rng.standard_normal(self.N))
        frf = FRF(method='ls_fir')
        freqs, h, g = frf.compute_lsfir(y, u, self.fs, self.m, self.N,
                                        nperseg=2048)
        return frf, freqs, h, g

    def _in_band_db_error(self, freqs, h):
        import scipy.signal as sig
        _, ht = sig.freqz(self.h_true, worN=freqs, fs=self.fs)
        band = (freqs >= 100.0) & (freqs <= 20000.0)
        return float(np.max(np.abs(20 * np.log10(np.abs(h[band]))
                                   - 20 * np.log10(np.abs(ht[band])))))

    def test_the_impulse_response_keeps_the_scale_of_the_channel_it_fits(self):
        # The true channel peaks at 1.0; the LU solve of the same equations
        # returns a peak of 16.9 on this record. The warning is pinned on its
        # own below, so this asserts the coefficients and nothing else.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, _, g = self._fit()
        assert np.max(np.abs(g)) == pytest.approx(1.0, abs=0.05)

    def test_the_frequency_response_matches_the_channel_across_the_swept_band(self):
        # The LU solve of the same equations is 46.6 dB out here.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, freqs, h, _ = self._fit()
        assert self._in_band_db_error(freqs, h) < 0.05

    def test_the_warning_names_the_order_and_the_condition_estimate(self):
        with pytest.warns(UserWarning, match=r"FIR order 512 is numerically "
                                             r"singular \(reciprocal condition "
                                             r"number "):
            frf, _, _, _ = self._fit()
        assert frf.info_rcond < np.finfo(float).eps

    #: Agreement demanded between the LU branch and a direct
    #: ``np.linalg.solve`` on a well-conditioned system, as a multiple of eps
    #: times the peak coefficient. numpy and scipy ship separate OpenBLAS
    #: builds, so the two run the same LAPACK algorithm from different
    #: binaries and bit-equality is a property of one machine's pairing, not
    #: of this code: measured over 60 well-conditioned solves it holds in 30
    #: and the worst disagreement is 2.75 eps of the peak. The fixtures below
    #: keep ``rcond`` above 0.05, where the LU's own backward error bounds the
    #: disagreement at roughly 20 eps, so this leaves 6x over the theory and
    #: 46x over the measurement.
    LU_AGREEMENT_EPS = 128.0

    @pytest.mark.parametrize("order", [64, 128, 256])
    def test_a_white_probe_takes_the_lu_branch_and_warns_about_nothing(self, order):
        rng = np.random.default_rng(11)
        u = rng.standard_normal(self.N)
        y = (np.convolve(u, self.h_true)[:self.N]
             + 1e-8 * rng.standard_normal(self.N))
        frf = FRF(method='ls_fir')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, _, g = frf.compute_lsfir(y, u, self.fs, order, self.N,
                                        nperseg=2048)
        assert frf.info_rcond > 0.05
        # A well-conditioned fit is the LU solution of the normal equations,
        # to the tolerance two builds of the same LAPACK routine can differ by.
        g_lu = np.linalg.solve(frf.Minfo, frf.Vinfo)
        tol = self.LU_AGREEMENT_EPS * np.finfo(float).eps * np.max(np.abs(g_lu))
        assert np.max(np.abs(g - g_lu)) <= tol


class TestLsFirInfoRcondFloorBoundary:
    """``_solve_info_matrices`` switches to the minimum-norm solution exactly
    at ``rcond <= _INFO_RCOND_FLOOR``. The fixtures are diagonal, where
    LAPACK's 1-norm reciprocal condition estimate is the smallest diagonal
    entry exactly, so the two sides of the threshold are reached by
    construction rather than by a fit that happens to land there.
    """

    n = 8

    def _system(self, delta):
        d = np.ones(self.n)
        d[-1] = delta
        return np.diag(d), np.ones(self.n)

    def test_above_the_floor_the_lu_coefficients_are_returned(self):
        from uacpy.acoustic_signal.system_id import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        delta = 2.0 * _INFO_RCOND_FLOOR
        minfo, vinfo = self._system(delta)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond == pytest.approx(delta, rel=1e-12)
        assert rcond > _INFO_RCOND_FLOOR
        assert g[-1] == pytest.approx(1.0 / delta, rel=1e-12)

    def test_at_the_floor_the_ill_conditioned_direction_is_dropped(self):
        from uacpy.acoustic_signal.system_id import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        delta = _INFO_RCOND_FLOOR
        minfo, vinfo = self._system(delta)
        with pytest.warns(UserWarning, match="numerically singular"):
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond == pytest.approx(delta, rel=1e-12)
        assert rcond <= _INFO_RCOND_FLOOR
        assert g[-1] == 0.0
        # The directions the product can still represent are untouched.
        assert g[:-1] == pytest.approx(np.ones(self.n - 1))

    def test_below_the_floor_the_ill_conditioned_direction_is_dropped(self):
        from uacpy.acoustic_signal.system_id import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        minfo, vinfo = self._system(0.5 * _INFO_RCOND_FLOOR)
        with pytest.warns(UserWarning, match="numerically singular"):
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond < _INFO_RCOND_FLOOR
        assert g[-1] == 0.0

    @pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
    def test_the_branch_does_not_move_with_the_amplitude_scale(self, scale):
        """A record in Pa and the same record in uPa must be fitted the same
        way: the threshold is on a reciprocal condition number, which both
        norms scale out of."""
        from uacpy.acoustic_signal.system_id import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        minfo, vinfo = self._system(2.0 * _INFO_RCOND_FLOOR)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, rcond = _solve_info_matrices(scale * minfo, scale * vinfo,
                                            self.n)
        assert rcond == pytest.approx(2.0 * _INFO_RCOND_FLOOR, rel=1e-12)

    def test_the_floor_sits_where_the_lu_error_bound_reaches_the_answer(self):
        """``cond(Minfo) * eps >= 1`` is the point at which the LU solution's
        error bound is the size of the answer itself, so the floor is
        float64's eps and not a tuned constant. The fixtures above move with
        the constant; this pins where the constant is."""
        from uacpy.acoustic_signal.system_id import _INFO_RCOND_FLOOR
        assert _INFO_RCOND_FLOOR == np.finfo(float).eps

    def test_an_exactly_singular_system_raises_the_error_the_order_search_skips_on(self):
        from uacpy.acoustic_signal.system_id import _solve_info_matrices
        with pytest.raises(np.linalg.LinAlgError):
            _solve_info_matrices(np.zeros((4, 4)), np.zeros(4), 4)


class TestFrfPublishesTheConditioningOfTheFitItReturns:
    def test_info_rcond_is_none_until_an_ls_fir_run_and_after_a_welch_one(self):
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        frf = FRF()
        assert frf.info_rcond is None
        frf.compute(u, y, 1000.0, method='ls_fir', m=8)
        assert 0.0 < frf.info_rcond <= 1.0
        frf.compute(u, y, 1000.0, method='welch', nperseg=512)
        assert frf.info_rcond is None


class TestLfmChirpRefusesNegativeSweepBounds:
    @pytest.mark.parametrize("fmin,fmax", [(-500.0, 1000.0), (1000.0, -500.0),
                                           (NAN, 1000.0)])
    def test_a_bound_below_zero_raises(self, fmin, fmax):
        with pytest.raises(ConfigurationError, match="finite frequency >= 0"):
            lfm_chirp(fmin, fmax, 0.1, 10000.0)

    def test_a_zero_start_frequency_is_accepted(self):
        _t, s = lfm_chirp(0.0, 1000.0, 0.1, 10000.0)
        assert np.all(np.isfinite(s))


def _flat_psd():
    frequencies = np.logspace(1.0, 3.0, 16)
    return 1e-6 / (1.0 + (frequencies / 100.0) ** 2), frequencies


def test_small_n_fft_warning_names_the_default_it_falls_back_to():
    """The behaviour is deliberate and documented: below 16 the argument is
    discarded for the 65536 default, not clamped up to 16. The message has to
    say that, or a reader takes it as ``n_fft=16``."""
    Pxx, frequencies = _flat_psd()
    with pytest.warns(UserWarning,
                      match=r'below the minimum 16; using the default 65536'):
        _, small, _ = synthesize_noise_from_psd(
            Pxx, frequencies, duration=0.05, n_fft=8, sample_rate=4000.0,
            rng=np.random.default_rng(0))
    _, default, _ = synthesize_noise_from_psd(
        Pxx, frequencies, duration=0.05, n_fft=65536, sample_rate=4000.0,
        rng=np.random.default_rng(0))
    np.testing.assert_array_equal(small, default)


class TestBpskModulateBipolarChips:
    def test_zero_one_bits_raise_a_typed_error(self):
        with pytest.raises(ConfigurationError, match='chip'):
            bpsk_modulate(np.array([0, 1, 1, 0]), 100.0, 1000.0, 100.0)

    def test_bipolar_chips_emit_one_signed_carrier_block_per_chip(self):
        chips = np.array([1, -1, 1, 1, -1, 1])
        fc, fs, cps = 100.0, 1000.0, 100.0
        s = bpsk_modulate(chips, fc, fs, cps)
        tone = np.sin(2 * np.pi * fc * np.arange(int(fs / cps)) / fs)
        np.testing.assert_allclose(
            s, np.concatenate([c * tone for c in chips]), atol=1e-12)


class TestSelectedOrderContract:
    def _fir_records(self, rows, seed=1):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((rows, 600))
        g = np.array([1.0, -0.5, 0.25])
        y = np.stack([np.convolve(u[i], g)[:600] for i in range(rows)])
        return u, y + 0.01 * rng.standard_normal(y.shape)

    def test_two_dimensional_input_publishes_one_order_per_row(self):
        u, y = self._fir_records(rows=3)
        frf = FRF()
        frf.compute(u, y, 1000.0, method="ls_fir", m="AIC", m_max=40)
        assert frf.selected_order == [3, 3, 3]

    def test_one_dimensional_criterion_input_publishes_an_int(self):
        u, y = self._fir_records(rows=1)
        frf = FRF()
        frf.compute(u[0], y[0], 1000.0, method="ls_fir", m="BIC", m_max=40)
        assert isinstance(frf.selected_order, int)
        assert frf.selected_order == 3

    def test_explicit_order_publishes_no_selected_order(self):
        u, y = self._fir_records(rows=1)
        frf = FRF()
        frf.compute(u[0], y[0], 1000.0, method="ls_fir", m=5)
        assert frf.selected_order is None
        assert len(np.asarray(frf.g)) == 5


class TestWelchMasksDegenerateDenominators:
    def test_zero_input_masks_h1_and_coherence_to_nan_with_warning(self):
        rng = np.random.default_rng(0)
        frf = FRF()
        with pytest.warns(UserWarning, match="compute_welch"):
            _, tf = frf.compute(np.zeros(2048), rng.standard_normal(2048),
                                1000.0, method="welch", nperseg=256)
        assert np.isnan(tf).all()
        assert np.isnan(frf.coh).all()

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
            frf1 = FRF()
            _, tf1 = frf1.compute(x, y, 1000.0, method="welch", nperseg=512)
            frf2 = FRF(estimator="H2")
            _, tf2 = frf2.compute(x, y, 1000.0, method="welch", nperseg=512)
        assert np.isfinite(tf1).all() and np.isfinite(tf2).all()
        assert np.isfinite(frf1.coh).all()


class TestSampleRateGuards:
    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_psd_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="psd: sample_rate must be > 0 Hz and finite"):
            psd(np.ones(64), bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_ppsd_rejects_bad_sample_rate(self, bad):
        with pytest.raises(ConfigurationError,
                           match="ppsd: sample_rate must be > 0 Hz"):
            ppsd(np.ones(64), bad, seg_duration=0.1)

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
    def test_sel_rejects_nonpositive_or_nonfinite_rate(self, bad):
        # fs=inf reached nfft = int(sample_rate) and raised OverflowError.
        with pytest.raises(ConfigurationError,
                           match="sel: sample_rate must be > 0 Hz and "
                                 "finite"):
            sel(np.ones(256), bad)

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
            bpsk_modulate(np.array([1, -1]), 100.0, bad, 100.0)

    def test_bpsk_modulate_rejects_infinite_chip_rate(self):
        # chips_per_sec=inf gave samples_per_chip = 0 and a silently empty
        # waveform.
        with pytest.raises(ConfigurationError,
                           match="bpsk_modulate: chips_per_sec must be > 0 "
                                 "chips/s and finite"):
            bpsk_modulate(np.array([1, -1]), 100.0, 1000.0, np.inf)


class TestAddNoiseMatchesTheRecordLengthExactly:
    """``add_noise`` draws its noise by sample count, not by a duration.

    ``int((n/fs)*fs)`` is ``n - 1`` for 7.02 % of lengths at 44100 Hz, 6.14 %
    at 48000 Hz and 5.16 % at 9600 Hz — always one short, never one long — so
    a length routed through seconds and back produces a noise vector the
    record cannot be added to.
    """

    @pytest.mark.parametrize('n, fs', [
        (1000, 8000.0), (1001, 8000.0), (5008, 9600.0), (44100, 44100.0),
        (30011, 48000.0),
    ])
    @pytest.mark.parametrize('n_rcv', [None, 4])
    def test_the_output_has_the_input_length(self, n, fs, n_rcv):
        x = np.zeros(n) if n_rcv is None else np.zeros((n, n_rcv))
        out = np.asarray(add_noise(x, fs, 0.0, 40.0, 1000.0, 500.0,
                                   rng=np.random.default_rng(0)))
        assert out.shape == x.shape

    @pytest.mark.parametrize('n, fs', [(1001, 8000.0), (5008, 9600.0)])
    def test_the_lengths_this_covers_are_the_ones_seconds_would_lose(
            self, n, fs):
        assert int((n / fs) * fs) == n - 1

    def test_a_record_shorter_than_the_filter_padding_is_named(self):
        with pytest.raises(ConfigurationError) as exc:
            add_noise(np.zeros(20), 8000.0, 0.0, 40.0, 1000.0, 500.0,
                      rng=np.random.default_rng(0))
        message = str(exc.value)
        assert 'add_noise' in message
        assert '20 sample(s)' in message

    def test_the_duration_taking_entry_point_sizes_by_duration(self):
        _, noise = make_bandlimited_noise(1000.0, 500.0, 1.0, 8000.0,
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
            sel(data, fs, integration_time=bad)

    @pytest.mark.parametrize('good, n_expected', [(4.0, 4000), (2.0, 2000)])
    def test_a_positive_value_truncates_from_the_start(
            self, good, n_expected):
        data, fs = self._record()
        got = np.nansum(sel(data, fs, integration_time=good).sel_pa2s)
        want = np.nansum(sel(data[:n_expected], fs).sel_pa2s)
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_smallest_admissible_value_is_one_sample_of_data(self):
        # The boundary the guard leaves alone: any value > 0 is accepted, and
        # the emptiness it can still produce is the other guard's message.
        data, fs = self._record()
        with pytest.raises(ConfigurationError, match='no samples to integrate'):
            sel(data, fs, integration_time=1e-9)


class TestFRFComputeKeywordsApplyToOneCallOnly:
    """A per-call ``method=`` / ``estimator=`` / ``nperseg=`` configures that
    run and nothing after it.

    The sharpest face is the frequency grid: one
    ``compute_periodic_etfe(nperseg=256)`` on a default ``FRF`` would move
    every later plain ``compute()`` onto a 129-bin axis instead of 4097 — a
    32x change in the axis two results are compared on, from a call that
    returned its own result and looked finished.
    """

    @staticmethod
    def _signals(n=16384):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        return x, np.convolve(x, np.ones(5) / 5)[:n]

    def test_a_method_override_does_not_stick(self):
        x, y = self._signals()
        frf = FRF()
        freqs_default, _ = frf.compute(x, y, 8000.0)
        frf.compute(x, y, 8000.0, method='etfe')
        assert frf.method == 'welch'
        freqs_after, _ = frf.compute(x, y, 8000.0)
        assert freqs_after.size == freqs_default.size

    def test_an_estimator_override_does_not_stick(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0, estimator='H2')
        assert frf.estimator == 'H1'

    @pytest.mark.parametrize('key, value', [('nperseg', 1024),
                                            ('noverlap', 64)])
    def test_a_welch_parameter_override_does_not_stick(self, key, value):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0, **{key: value})
        assert frf.params[key] == FRF().params[key]

    def test_compute_periodic_etfe_does_not_move_the_shared_grid(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute_periodic_etfe(x, y, 8000.0, nperseg=256)
        assert frf.params['nperseg'] == FRF().params['nperseg']
        freqs, _ = frf.compute(x, y, 8000.0)
        assert freqs.size == FRF().params['nperseg'] // 2 + 1

    def test_compute_lsfir_does_not_move_the_shared_grid(self):
        rng = np.random.default_rng(4)
        u = rng.standard_normal(600)
        y = np.convolve(u, np.array([1.0, -0.5, 0.25]))[:u.size]
        frf = FRF()
        frf.compute_lsfir(y, u, 1000.0, m=8, N=600, nperseg=128)
        assert frf.params['nperseg'] == FRF().params['nperseg']

    def test_the_override_reaches_the_run_it_was_given_to(self):
        """The negative half: a per-call keyword must change *this* result."""
        x, y = self._signals()
        frf = FRF()
        freqs_default, _ = frf.compute(x, y, 8000.0)
        freqs_override, _ = frf.compute(x, y, 8000.0, nperseg=1024)
        assert freqs_default.size == 4097
        assert freqs_override.size == 513

    def test_the_result_attributes_are_rewritten_by_every_run(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0)
        assert frf.coh is not None
        frf.compute(x, y, 8000.0, method='etfe')
        assert frf.coh is None
        assert frf.frequencies.size == x.size // 2 + 1


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
         lambda f, fs: tone_burst(f, 5, fs)),
        ('lfm_chirp',
         lambda f, fs: lfm_chirp(100.0, f, 0.1, fs)),
        ('hfm_chirp',
         lambda f, fs: hfm_chirp(100.0, f, 0.1, fs)),
        ('bpsk_modulate',
         lambda f, fs: bpsk_modulate(np.array([1, -1, 1, -1]), f, fs, 100.0)),
        ('fsk_modulate',
         lambda f, fs: _fsk_modulate(np.array([0, 1]), np.array([1000.0, f]),
                                     0.01, fs)),
        ('fsk_demodulate',
         lambda f, fs: _fsk_demodulate(np.zeros(1000),
                                       np.array([1000.0, f]), 0.01, fs)),
        ('make_noise_waveform',
         lambda f, fs: make_noise_waveform(f - 100.0, 200.0, 1.0, fs,
                                           rng=np.random.default_rng(0))),
    ])
    def test_a_generator_refuses_exactly_nyquist_and_accepts_just_below(
            self, name, call):
        fs = self.FS
        with pytest.raises(ConfigurationError, match='Nyquist'):
            call(fs / 2, fs)
        call(fs / 2 - 1.0, fs)

    @pytest.mark.parametrize('bad', [NAN, np.inf])
    def test_a_generator_refuses_a_non_finite_frequency(self, bad):
        with pytest.raises(ConfigurationError, match='Nyquist'):
            make_noise_waveform(bad, 200.0, 1.0, self.FS,
                                rng=np.random.default_rng(0))

    def test_make_noise_waveform_refuses_a_band_straddling_dc(self):
        with pytest.raises(ConfigurationError, match='lower band edge'):
            make_noise_waveform(-1000.0, 200.0, 1.0, self.FS,
                                rng=np.random.default_rng(0))

    @pytest.mark.parametrize('fc', [1000.0, 2000.0, 4000.0])
    def test_an_admitted_band_lands_where_it_was_asked_for(self, fc):
        """The negative half of the guard: below Nyquist the band is where the
        caller put it, so the guard is refusing folds and nothing else."""
        _, x = make_noise_waveform(fc, 200.0, 1.0, self.FS,
                                   rng=np.random.default_rng(0))
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.FS)
        peak = freqs[int(np.argmax(np.abs(np.fft.rfft(x))))]
        assert abs(peak - fc) < 150.0

    def test_the_analyser_side_admits_exactly_nyquist_and_refuses_above(self):
        from uacpy.acoustic_signal import constant_q_psd
        fs = self.FS
        x = np.random.default_rng(0).standard_normal(8000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            constant_q_psd(x, fs, fmin=100.0, fmax=fs / 2)
        with pytest.raises(ConfigurationError, match='Nyquist'):
            constant_q_psd(x, fs, fmin=100.0, fmax=fs / 2 + 1.0)

    def test_the_two_helpers_disagree_only_at_exactly_nyquist(self):
        from uacpy.acoustic_signal._signal_validate import (
            require_at_most_nyquist, require_below_nyquist)
        fs = self.FS
        require_below_nyquist(fs / 2 - 1e-9, fs, 'probe', 'f', 'it aliases')
        require_at_most_nyquist(fs / 2, fs, 'probe', 'f', 'it aliases')
        with pytest.raises(ConfigurationError):
            require_below_nyquist(fs / 2, fs, 'probe', 'f', 'it aliases')
        with pytest.raises(ConfigurationError):
            require_at_most_nyquist(fs / 2 + 1e-9, fs, 'probe', 'f',
                                    'it aliases')

    def test_an_array_valued_guard_names_the_offending_frequencies(self):
        from uacpy.acoustic_signal._signal_validate import (
            require_below_nyquist)
        with pytest.raises(ConfigurationError) as exc:
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
        ('constant_q_psd',
         lambda d, b: _constant_q_psd(d, b, fmin=20.0, fmax=100.0)),
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
    @pytest.mark.parametrize('arg', ['range_m', 'c'])
    def test_warp_signal_guards_its_range_and_sound_speed(self, arg, bad):
        kwargs = {'range_m': 1000.0}
        if arg == 'c':
            kwargs['c'] = bad
        else:
            kwargs['range_m'] = bad
        with pytest.raises(ConfigurationError, match=arg):
            _warp_signal(self.DATA_1D, 1000.0, **kwargs)

    def test_wigner_ville_axes_are_finite_on_an_accepted_rate(self):
        """The negative control: a valid rate still returns usable axes, so
        the guard is refusing the garbage cases and nothing else."""
        result = _wigner_ville(self.DATA_1D, 1000.0)
        assert np.all(np.isfinite(result.frequencies))
        assert np.all(np.isfinite(result.times))
        assert np.all(np.diff(result.frequencies) > 0)


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
        with pytest.raises(ConfigurationError) as exc:
            psd(self._trace(), 1600.0)
        message = str(exc.value)
        assert 'psd:' in message
        assert 'Field' in message
        assert 'Field.data' in message

    def test_the_bridge_the_message_names_actually_works(self):
        trace = self._trace()
        freqs, power = psd(np.asarray(trace.data), 1600.0)
        assert freqs.size == power.size


class TestSignalAxisGuardsRefuseAnEmptyAxis:
    """Three entry points validated axis monotonicity but not axis emptiness,
    so an empty frequency axis reached numpy and raised an untyped
    ``ValueError`` / ``IndexError`` naming no input the caller supplied — the
    exact failure the canonical guard in ``core._carrier_validate`` says it
    exists to prevent.

    The monotonicity predicate itself agrees across all five copies of it in
    the package; only the empty and one-sample corners differed.
    """

    @staticmethod
    def _calls():
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function)
        from uacpy.acoustic_signal.modal import modal_group_velocity
        return {
            # A monotonic k_horizontal: a propagating mode's wavenumber
            # rises with frequency, and a flat one is refused (it divides the
            # group velocity by zero), so a constant axis would not exercise
            # the well-formed path this helper is shared with.
            'modal_group_velocity':
                lambda f, n: modal_group_velocity(f, np.linspace(0.5, 12.0, n)),
            'impulse_response_from_transfer_function':
                lambda f, n: impulse_response_from_transfer_function(
                    np.ones(n, dtype=complex), f, 1000.0),
            'decidecade_band_levels':
                lambda f, n: decidecade_band_levels(np.ones(n), f),
        }

    @pytest.mark.parametrize('name', ['modal_group_velocity',
                                      'impulse_response_from_transfer_function',
                                      'decidecade_band_levels'])
    def test_an_empty_axis_raises_a_typed_error_naming_the_axis(self, name):
        call = self._calls()[name]
        with pytest.raises(ConfigurationError) as exc:
            call(np.array([], dtype=float), 0)
        message = str(exc.value)
        assert name in message
        assert 'frequencies' in message
        assert 'at least one value' in message

    @pytest.mark.parametrize('name', ['modal_group_velocity',
                                      'decidecade_band_levels'])
    def test_a_one_sample_axis_names_its_own_domain_minimum(self, name):
        call = self._calls()[name]
        with pytest.raises(ConfigurationError) as exc:
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
        with pytest.raises(ConfigurationError) as exc:
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
        from uacpy.acoustic_signal.noise_synthesis import (
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


class TestMseqAndDsssSequencesAreNotInterchangeable:
    """``mseq`` and ``comms.dsss.m_sequence`` share the BPSK polarity but not
    the register seed, so despreading one's output with the other's sequence
    lands on the m-sequence's off-peak correlation ``-1/N`` rather than on the
    symbol — the property ``mseq``'s docstring states."""

    SYMBOLS = np.array([1.0, -1.0, 1.0])

    @staticmethod
    def _pair(n=5, taps=(5, 2)):
        from uacpy.acoustic_signal.sequences import mseq
        from uacpy.comms.dsss import m_sequence
        return mseq(n), m_sequence(n, list(taps))

    def test_the_same_generator_at_both_ends_recovers_the_symbols(self):
        from uacpy.comms.dsss import despread, spread
        for seq in self._pair():
            got = despread(spread(self.SYMBOLS, seq), seq)
            np.testing.assert_allclose(np.real(got), self.SYMBOLS, atol=1e-9)

    @pytest.mark.parametrize('taps', [(5, 2), (5, 3)])
    def test_crossing_the_two_generators_collapses_to_minus_one_over_n(
            self, taps):
        from uacpy.comms.dsss import despread, spread
        a, b = self._pair(5, taps)
        got = np.real(despread(spread(self.SYMBOLS, a), b))
        np.testing.assert_allclose(got, -self.SYMBOLS / a.size, atol=1e-9)

    def test_the_seeds_are_what_differ_not_the_polarity(self):
        a, b = self._pair()
        assert set(np.unique(a)) == {-1.0, 1.0}
        assert set(np.unique(np.asarray(b, dtype=float))) == {-1.0, 1.0}
        assert not np.array_equal(a, np.asarray(b, dtype=float))

    def test_even_the_same_cycle_is_only_a_shift_of_it(self):
        a, b = self._pair(5, (5, 2))
        b = np.asarray(b, dtype=float)
        xcorr = np.array([float(np.dot(a, np.roll(b, k)))
                          for k in range(a.size)])
        assert xcorr.max() == a.size          # same cycle
        assert xcorr[0] != a.size             # but not at zero lag


class TestEstimatorOutputDtypesMatchTheDocumentedSplit:
    """The package docstring states which estimators preserve a ``float32``
    input and which promote. Undocumented, the split surfaces as a silent
    promotion when two estimates of the same record are stacked."""

    FS = 1000.0

    @staticmethod
    def _record(dtype):
        return np.random.default_rng(0).standard_normal(4096).astype(dtype)

    @pytest.mark.parametrize('name, call, field', [
        ('psd', lambda x, fs: psd(x, fs), 'power'),
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
        ('constant_q_psd',
         lambda x, fs: _constant_q_psd(x, fs, fmin=20.0, fmax=400.0)),
        ('sel', lambda x, fs: sel(x, fs).sel_pa2s),
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
            assert psd(x, self.FS).power.dtype == np.float64
            assert spectrogram(x, self.FS).power.dtype == np.float64
            assert _envelope(x).dtype == np.float64


def test_the_signal_guide_states_the_per_call_frf_contract():
    """``docs/guide/signal.md``'s FRF section documented ``frf.m`` reading back
    as the criterion string, which is the behaviour the per-call resolution
    removed. A prose page cannot be pinned wholesale, so this pins the one
    sentence that went stale and the readback it demonstrated."""
    import pathlib
    page = (pathlib.Path(__file__).resolve().parents[2]
            / 'docs' / 'guide' / 'signal.md')
    if not page.is_file():
        pytest.skip('docs/ is not present (source checkout only)')
    text = page.read_text(encoding='utf-8')
    assert 'Every `compute` argument applies to that call alone' in text
    assert '**`m` holds the criterion' not in text
    # The snippet must not show the criterion coming back off the object.
    assert ">>> frf.m\n'CP'" not in text
    # And the readback the page does show has to be what the code returns.
    from uacpy.acoustic_signal.system_id import FRF
    rng = np.random.default_rng(2)
    n = 3000
    u = rng.standard_normal(n)
    g = rng.standard_normal(6)
    g = g / np.linalg.norm(g)
    clean = np.convolve(u, g)[:n]
    y = clean + 0.1 * np.std(clean) * rng.standard_normal(n)
    frf = FRF(method='ls_fir')
    frf.compute(u, y, 1000.0, m='CP')
    assert frf.selected_order == 6
    assert frf.m == FRF(method='ls_fir').m == 512
