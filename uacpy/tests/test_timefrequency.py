"""Tests for ``uacpy.acoustic_signal``'s time-frequency transforms.

The analytic signal and its envelope and instantaneous frequency; the
continuous wavelet transform and its inverse; the Wigner-Ville distribution;
and the cepstrum family, real and complex.

Two properties recur and are what most of these pin. **Invertibility** — a
transform that claims an inverse must return the input, so ``inverse_cwt`` and
``inverse_complex_cepstrum`` are checked against round trips rather than
against a reference implementation. And **what the transform is entitled to
assume**: an axis that is not monotonic, a rate that is not positive, a window
longer than the record. Each is refused by name rather than answered with a
number that looks plausible.

The channel model and the modal warping that used to share this file now live
in ``test_channel.py`` and ``test_modal.py``.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    analytic_signal,
    cepstrum,
    complex_cepstrum,
    cwt,
    envelope,
    inverse_cwt,
    inverse_complex_cepstrum,
    instantaneous_frequency,
    wigner_ville,
)
from uacpy.core.exceptions import ConfigurationError

FS = 10000.0


class TestTimeFrequency:
    def test_analytic_signal_real_part(self):
        x = np.cos(2 * np.pi * 50 * np.arange(1000) / FS)
        z = analytic_signal(x)
        assert np.allclose(z.real, x, atol=1e-9)

    def test_envelope_of_tone_is_flat(self):
        x = np.cos(2 * np.pi * 200 * np.arange(2000) / FS)
        env = envelope(x)
        assert np.std(env[100:-100]) < 0.05

    def test_instantaneous_frequency_of_tone(self):
        f0 = 300.0
        x = np.cos(2 * np.pi * f0 * np.arange(2000) / FS)
        inst = instantaneous_frequency(x, FS)
        # np.gradient gives a centred, time-aligned estimate of length len(x).
        assert inst.shape == x.shape
        assert np.median(inst[50:-50]) == pytest.approx(f0, abs=2.0)

    def test_wigner_ville_localises_tone(self):
        f0 = 250.0
        x = np.cos(2 * np.pi * f0 * np.arange(256) / FS)
        f, t, W = wigner_ville(x, FS)
        peak_f = f[np.argmax(W.mean(axis=1))]
        assert peak_f == pytest.approx(f0, abs=FS / 256)

    def test_wigner_ville_nfft_grows_frequency_axis(self):
        x = np.cos(2 * np.pi * 250.0 * np.arange(256) / FS)
        f, t, W = wigner_ville(x, FS, nfft=512)
        assert W.shape == (512, 256) and f.size == 512
        assert f[np.argmax(W.mean(axis=1))] == pytest.approx(250.0, abs=FS / 512)

    def test_wigner_ville_smoothing_suppresses_cross_terms(self):
        # Two tones produce a spurious cross-term midway between them in the
        # plain WVD; smoothed-pseudo-WVD windows attenuate it.
        nfft_n = 256
        nn = np.arange(nfft_n)
        x = np.cos(2 * np.pi * 150.0 * nn / FS) + np.cos(2 * np.pi * 400.0 * nn / FS)
        f, _, W_plain = wigner_ville(x, FS)
        _, _, W_smooth = wigner_ville(x, FS, freq_window=65, time_window=33)
        mid = np.argmin(np.abs(f - 275.0))   # cross-term location
        band = slice(mid - 2, mid + 3)
        # Cross-terms oscillate in time, so use magnitude (not a time-mean that
        # would cancel them); smoothing damps that oscillation.
        cross_plain = np.abs(W_plain[band]).mean()
        cross_smooth = np.abs(W_smooth[band]).mean()
        assert cross_smooth < 0.5 * cross_plain

    def test_wigner_ville_axis_order_matches_siblings(self):
        """All three time-frequency results carry an ``(n_freq, n_time)``
        payload, so all three must name their axes in that order — otherwise a
        square distribution transposes silently."""
        from uacpy.acoustic_signal.constant_q import constant_q_spectrogram
        from uacpy.acoustic_signal.timefreq import (
            SpectrogramResult, WignerVilleResult, spectrogram)
        assert (WignerVilleResult._fields[:2]
                == SpectrogramResult._fields[:2] == ("frequencies", "times"))
        x = np.cos(2 * np.pi * 250.0 * np.arange(1024) / FS)
        for res in (wigner_ville(x[:256], FS), spectrogram(x, FS, nperseg=256),
                    constant_q_spectrogram(x, FS, fmin=200.0, fmax=2000.0)):
            assert res[2].shape == (res.frequencies.size, res.times.size)

    def test_wigner_ville_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            wigner_ville(np.zeros(256), FS, nfft=128)

    def test_wigner_ville_time_marginal_and_energy(self):
        # Defining WVD property (convention-independent): sum_f W(t,f) is
        # exactly proportional to the instantaneous power |z(t)|^2, and the
        # total integrates to the signal energy — validates the normalisation.
        n = 256
        t = np.arange(n) / FS
        x = np.cos(2 * np.pi * 0.1 * FS * t) * np.exp(-((t - t[n // 3]) / 0.04) ** 2)
        power = np.abs(analytic_signal(x)) ** 2
        _, _, W = wigner_ville(x, FS)
        m = power > 1e-6 * power.max()
        ratio = W.sum(axis=0)[m] / power[m]
        assert np.allclose(ratio, n, rtol=1e-9)            # constant = N exactly
        assert np.isclose(W.sum(), n * power.sum(), rtol=1e-9)   # energy

    def test_spwvd_smoothing_preserves_invariants(self):
        # Signal localised to the window centre so the time-smoothing window is
        # never edge-clipped (its energy conservation is then exact).
        n = 256
        t = np.arange(n) / FS
        x = np.cos(2 * np.pi * 0.12 * FS * t) * np.exp(-((t - t[n // 2]) / 0.0025) ** 2)
        power = np.abs(analytic_signal(x)) ** 2
        m = power > 1e-6 * power.max()
        _, _, W = wigner_ville(x, FS)
        # pseudo-WVD: the lag window has h(0)=1, so the time marginal is intact
        _, _, Wp = wigner_ville(x, FS, freq_window=65)
        assert np.allclose(Wp.sum(axis=0)[m] / power[m], n, rtol=1e-9)
        # smoothed-pseudo-WVD: the (Σg-normalised) time window conserves energy
        _, _, Ws = wigner_ville(x, FS, freq_window=65, time_window=33)
        assert np.isclose(Ws.sum(), W.sum(), rtol=1e-9)

    def test_cepstrum_finite(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(512)
        assert np.all(np.isfinite(cepstrum(x)))
        assert np.all(np.isfinite(complex_cepstrum(x).cepstrum))

    def test_complex_cepstrum_round_trip(self):
        # Smooth signal: phase unwraps cleanly so the homomorphic round trip
        # is near-exact (white noise is the pathological case for unwrap).
        t = np.arange(256) / FS
        x = np.sin(2 * np.pi * 60 * t) * np.exp(-0.5 * ((t - 0.12) / 0.04) ** 2)
        rec = inverse_complex_cepstrum(complex_cepstrum(x))
        assert np.max(np.abs(x - rec)) < 1e-8

    def test_cepstrum_echo_peak(self):
        # An echo at delay D shows a cepstral peak at quefrency D.
        rng = np.random.default_rng(0)
        n = 2048
        x = rng.standard_normal(n)
        d = 120
        x[d:] += 0.8 * x[:-d]
        c = cepstrum(x)
        lo, hi = 20, n // 2
        assert lo + np.argmax(c[lo:hi]) == pytest.approx(d, abs=1)

    def test_cepstrum_window_nfft_lifter(self):
        rng = np.random.default_rng(1)
        n = 512
        x = rng.standard_normal(n)
        c = cepstrum(x, window="hann", nfft=1024)
        assert c.size == 1024 and np.all(np.isfinite(c))
        # Long-pass lifter zeros the low quefrencies (spectral envelope).
        lifted = cepstrum(x, lifter=-10)
        assert np.allclose(lifted[:10], 0.0) and lifted[0] == 0.0

    def test_cepstrum_echo_survives_longpass_lifter(self):
        rng = np.random.default_rng(0)
        n = 2048
        x = rng.standard_normal(n)
        d = 120
        x[d:] += 0.8 * x[:-d]
        c = cepstrum(x, lifter=-30)  # remove smooth envelope, keep echo peak
        lo, hi = 40, n // 2
        assert lo + np.argmax(c[lo:hi]) == pytest.approx(d, abs=1)

    def test_cepstrum_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            cepstrum(np.zeros(512), nfft=256)


class TestCWT:
    @pytest.mark.parametrize("wavelet", ["morlet", "paul", "dog"])
    def test_localizes_tone(self, wavelet):
        x = np.cos(2 * np.pi * 200 * np.arange(2048) / FS)
        freqs, W = cwt(x, FS, wavelet=wavelet)
        ridge = freqs[np.argmax(np.abs(W).mean(axis=1))]
        assert ridge == pytest.approx(200.0, rel=0.08)

    def test_shape_and_explicit_freqs(self):
        x = np.cos(2 * np.pi * 100 * np.arange(1024) / FS)
        freqs = np.array([50.0, 100.0, 200.0])
        f, W = cwt(x, FS, frequencies=freqs)
        assert W.shape == (3, 1024)
        assert np.iscomplexobj(W)

    def test_bad_wavelet_raises(self):
        with pytest.raises(ConfigurationError):
            cwt(np.zeros(128), FS, wavelet="haar")

    @pytest.mark.parametrize("wavelet", ["morlet", "paul", "dog"])
    def test_icwt_round_trip_shape(self, wavelet):
        t = np.arange(512) / FS
        x = (np.sin(2 * np.pi * 60 * t) * np.exp(-0.5 * ((t - 0.25) / 0.05) ** 2)
             + 0.4 * np.sin(2 * np.pi * 150 * t))
        f, W = cwt(x, FS, wavelet=wavelet, n_freqs=96)
        xr = inverse_cwt(W, f, FS, wavelet=wavelet)
        # A 96-scale bank over a finite frequency span cannot resolve the
        # identity, and the cone of influence corrupts both ends of the record,
        # so 0.95 is a shape check across all three wavelets rather than a
        # reconstruction bound. The amplitude fidelity that IS pinned lives in
        # test_icwt_amplitude_holds_for_nondefault_orders, which measures the
        # interior only.
        assert np.corrcoef(x, xr)[0, 1] > 0.95

    def test_icwt_bad_shape_raises(self):
        with pytest.raises(ConfigurationError):
            inverse_cwt(np.zeros((3, 10)), np.array([1.0, 2.0]), FS)

    @pytest.mark.parametrize("wavelet,kw", [
        ("morlet", {"w0": 4.0}), ("morlet", {"w0": 6.0}), ("morlet", {"w0": 12.0}),
        ("paul", {"order": 2}), ("paul", {"order": 4}), ("paul", {"order": 8}),
        ("dog", {"order": 2}), ("dog", {"order": 4}), ("dog", {"order": 8}),
    ])
    def test_icwt_amplitude_holds_for_nondefault_orders(self, wavelet, kw):
        """``inverse_cwt`` must reconstruct at the right amplitude for whatever
        ``w0``/``order`` :func:`cwt` was run with — pinned reconstruction
        constants mis-scale non-default orders by up to 2x and, for DOG m=4,
        invert the sign."""
        n = 2048
        t = np.arange(n) / FS
        x = (1.7 * np.sin(2 * np.pi * 450.0 * t)
             + 0.8 * np.sin(2 * np.pi * 1500.0 * t))
        f, W = cwt(x, FS, wavelet=wavelet, n_freqs=200, **kw)
        xr = inverse_cwt(W, f, FS, wavelet=wavelet, **kw)
        mid = slice(n // 4, 3 * n // 4)      # outside the cone of influence
        assert np.std(xr[mid]) / np.std(x[mid]) == pytest.approx(1.0, rel=0.06)

    @pytest.mark.parametrize("wavelet,w0,order,c_delta,psi0", [
        ("morlet", 6.0, 2, 0.776, 0.7511),
        ("paul", 6.0, 4, 1.132, 1.079),
        ("dog", 6.0, 2, 3.541, 0.867),
    ])
    def test_reconstruction_constants_match_torrence_compo(
            self, wavelet, w0, order, c_delta, psi0):
        """The derived constants must agree with Torrence & Compo (1998)
        Table 2 at the three tabulated orders.

        Theory-vs-numeric tolerances, not float noise. ``_reconstruction_constants``
        derives both by quadrature while T&C's caption calls the table
        "empirically derived", so the two disagree by a real, wavelet-dependent
        amount: C_delta is out by 0.31 % (Morlet), 0.10 % (Paul) and 2.13 %
        (DOG m=2, 3.616 derived vs 3.541 tabulated) — rel=0.03 is sized by the
        DOG case. psi0(0) is a closed-form integral and matches to <4e-4, so
        rel=0.01 only has to cover T&C's 4-significant-figure rounding.
        """
        from uacpy.acoustic_signal.timefreq import _reconstruction_constants
        cd, p0 = _reconstruction_constants(wavelet, w0, order)
        assert cd == pytest.approx(c_delta, rel=0.03)
        assert p0 == pytest.approx(psi0, rel=0.01)

    @pytest.mark.parametrize("order", [1, 3, 5])
    def test_icwt_odd_dog_order_raises(self, order):
        """An odd DOG is an odd function: psi0(0) = 0 and the eq.-11 inverse
        does not exist. Reject it instead of dividing by a pinned constant."""
        f, W = cwt(np.zeros(512), FS, wavelet="dog", order=order, n_freqs=16)
        with pytest.raises(ConfigurationError):
            inverse_cwt(W, f, FS, wavelet="dog", order=order)


class TestComplexCepstrumRemovesLinearPhase:
    """The unwrapped phase of a delayed signal carries a linear ramp whose
    inverse transform is a ``1/q`` tail. Left in, it swamps the echo structure
    the cepstrum exists to show — an echo at quefrency 120 measured 0.195
    against a tail of 100 at q=1 — and makes the result depend on the
    signal's absolute arrival time rather than on its echo delays."""

    @staticmethod
    def _two_arrivals(n=1024, t0=50, gap=120, amp=0.6):
        x = np.zeros(n)
        x[t0] = 1.0
        x[t0 + gap] = amp
        return x

    def test_echo_dominates_the_tail(self):
        from uacpy.acoustic_signal import complex_cepstrum
        c = np.asarray(complex_cepstrum(self._two_arrivals()).cepstrum)
        assert abs(c[120]) == pytest.approx(0.6, abs=1e-6)
        assert abs(c[1]) < 1e-9          # the 1/q tail is gone

    def test_result_is_independent_of_arrival_time(self):
        # The discriminating property: the cepstrum describes echo delays, so
        # shifting the whole signal must not change it.
        from uacpy.acoustic_signal import complex_cepstrum
        a = np.asarray(complex_cepstrum(self._two_arrivals(t0=50)).cepstrum)
        b = np.asarray(complex_cepstrum(self._two_arrivals(t0=200)).cepstrum)
        assert np.max(np.abs(a - b)) < 1e-9

    def test_round_trip_is_exact(self):
        from uacpy.acoustic_signal import (complex_cepstrum,
                                           inverse_complex_cepstrum)
        x = self._two_arrivals()
        assert np.max(np.abs(x - inverse_complex_cepstrum(
            complex_cepstrum(x)))) < 1e-9
