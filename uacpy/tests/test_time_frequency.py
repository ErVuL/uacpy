"""Time-frequency transforms in ``uacpy.acoustic_signal.estimate``.

The analytic signal and its envelope and instantaneous frequency; the
spectrogram; the continuous wavelet transform and its inverse; the
Wigner-Ville distribution; and the cepstrum family, real and complex.

Three things are pinned here. **Agreement** — the spectrogram is checked
against scipy at matched settings, so a default that drifts apart shows up as
a number rather than as a shape. **Invertibility** — a transform that claims
an inverse must return the input, so ``inverse_cwt`` and
``inverse_complex_cepstrum`` are checked against round trips rather than
against a reference implementation. And **what the transform is entitled to
assume**: an axis that is not monotonic, a rate that is not positive, a window
longer than the record. Each is refused by name rather than answered with a
number that looks plausible.

The plotters that draw these transforms are exercised alongside them, since a
transform and the figure it produces share the axis conventions.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.signal import get_window as _get_window  # noqa: E402
from scipy.signal import spectrogram as _scipy_spec  # noqa: E402

from uacpy.acoustic_signal import (  # noqa: E402
    analytic_signal,
    cepstrum,
    complex_cepstrum,
    cwt,
    envelope,
    instantaneous_frequency,
    inverse_complex_cepstrum,
    inverse_cwt,
    spectrogram,
    wigner_ville,
)
from uacpy.acoustic_signal.estimate import _smoothing_window  # noqa: E402
from uacpy.core.exceptions import ConfigurationError  # noqa: E402
from uacpy.visualization.plots.signal import (  # noqa: E402
    plot_cepstrum, plot_cwt, plot_spectrogram, plot_wigner_ville)

FS = 10000.0


# ── round trips, refusals, and what each transform assumes ──────────────────

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
        from uacpy.acoustic_signal.estimate import constant_q_spectrogram
        from uacpy.acoustic_signal.estimate import (
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

    def test_wigner_ville_refuses_surface_past_cell_cap_before_allocating(
            self, monkeypatch):
        # (nfft, n) = (1 << 17, 2000) is 2.6e8 float64 cells, past the 1 << 27
        # ceiling ambiguity_function shares. The refusal must come before
        # np.zeros runs: the kernel backs the surface lazily, so a cap checked
        # after the allocation would still let the per-time loop swap the host
        # instead of raising.
        import uacpy.acoustic_signal.estimate as timefreq

        class AllocationAttempted(Exception):
            pass

        class NumpyRefusingZeros:
            def __getattr__(self, name):
                if name == "zeros":
                    def refuse(*args, **kwargs):
                        raise AllocationAttempted(args)
                    return refuse
                return getattr(np, name)

        monkeypatch.setattr(timefreq, "np", NumpyRefusingZeros())
        z = np.ones(2000, dtype=complex)
        with pytest.raises(ConfigurationError, match=r"131072 x 2000.*nfft"):
            wigner_ville(z, 48000.0, nfft=1 << 17)

    def test_wigner_ville_cell_cap_admits_equality_and_refuses_one_past(
            self, monkeypatch):
        import uacpy.acoustic_signal.estimate as timefreq
        n = 64
        monkeypatch.setattr(timefreq, "_MAX_WIGNER_CELLS", n * n)
        z = np.ones(n, dtype=complex)
        assert wigner_ville(z, FS, nfft=n).distribution.shape == (n, n)
        with pytest.raises(ConfigurationError, match=r"65 x 64 = 4160"):
            wigner_ville(z, FS, nfft=n + 1)

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

    @pytest.mark.parametrize("fs", [4000.0, 8000.0, 16000.0, 22050.0,
                                    44100.0, 48000.0, 96000.0])
    def test_default_grid_runs_from_four_cycles_to_exactly_nyquist(self, fs):
        # The default grid caps itself at fs/2, which the analyser rule admits
        # (require_at_most_nyquist); its endpoints are exact, not 10**log10.
        n = 1024
        x = np.cos(2 * np.pi * 0.1 * fs * np.arange(n) / fs)
        freqs, W = cwt(x, fs)
        assert freqs[-1] == fs / 2.0
        assert freqs[0] == 4.0 * fs / n
        assert W.shape == (freqs.size, n)

    def test_explicit_grid_admits_nyquist_and_refuses_one_ulp_past(self):
        x = np.cos(2 * np.pi * 100 * np.arange(256) / FS)
        nyq = FS / 2.0
        f, _ = cwt(x, FS, frequencies=[100.0, nyq])
        assert f[-1] == nyq
        with pytest.raises(ConfigurationError, match="above the Nyquist"):
            cwt(x, FS, frequencies=[100.0, np.nextafter(nyq, np.inf)])

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
        from uacpy.acoustic_signal.estimate import _reconstruction_constants
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


# ── agreement with scipy, and the plotters that draw these transforms ───────

def test_spectrogram_matches_scipy():
    x = np.random.default_rng(0).standard_normal(48000)
    # spectrogram passes noverlap=None straight through, letting scipy derive
    # nperseg//8; the reference must use the same default or it compares two
    # different overlaps.
    f0, t0, S0 = _scipy_spec(x, 48000.0, window="hann", nperseg=8192,
                             noverlap=None, scaling="density", mode="psd")
    f, t, S = spectrogram(x, 48000.0)
    assert np.allclose(f, f0) and np.allclose(S, S0)
    fig, ax = plot_spectrogram(f, t, S)
    assert ax.collections
    plt.close(fig)


def test_timefreq_plots():
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(1024) / fs)
    fr, W = cwt(x, fs, np.linspace(20, 200, 40))
    fig, ax = plot_cwt(fr, W, fs)
    assert ax.collections
    plt.close(fig)
    f, t, wv = wigner_ville(x[:256], fs)
    fig, ax = plot_wigner_ville(f, t, wv)
    plt.close(fig)
    c = cepstrum(x)
    fig, ax = plot_cepstrum(c, sample_rate=fs)
    assert ax.lines
    plt.close(fig)

def test_even_array_freq_window_keeps_transform_real():
    """An even-length array `freq_window` keeps its centre sample on-centre,
    so acc(-tau) = conj(acc(tau)) holds and the imaginary part the transform
    discards is pure rounding error."""
    fs = 1000.0
    x = np.sin(2 * np.pi * 100 * np.arange(256) / fs)
    even = _get_window("hann", 4, fftbins=False)
    hv, Lh = _smoothing_window(even, "freq_window")
    assert hv.size % 2 == 1
    np.testing.assert_allclose(hv, hv[::-1])
    # Measure the imaginary part wigner_ville's `.real` discards by rebuilding
    # its lag kernel with the same window.
    z = analytic_signal(x)
    n = z.size
    num = den = 0.0
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, Lh)
        taus = np.arange(-taumax, taumax + 1)
        kernel = np.zeros(n, dtype=complex)
        kernel[(taus + n) % n] = (z[ti + taus] * np.conj(z[ti - taus])
                                  * hv[Lh + taus])
        F = np.fft.fft(kernel)
        num += np.sum(np.imag(F) ** 2)
        den += np.sum(np.real(F) ** 2)
    assert np.sqrt(num / den) < 1e-12
    # And the public API accepts the even array, matching the explicit
    # centre-deleted odd window.
    Wa = wigner_ville(x, fs, freq_window=even)
    Wb = wigner_ville(x, fs, freq_window=np.delete(even, even.size // 2))
    np.testing.assert_allclose(Wa.distribution, Wb.distribution)


def test_smoothed_pseudo_wvd_matches_per_lag_reference():
    """The time-smoothed lag kernel equals a per-lag scalar-loop reference."""
    fs = 1000.0
    n = 96
    tt = np.arange(n) / fs
    x = np.sin(2 * np.pi * 80 * tt) + 0.5 * np.sin(2 * np.pi * 200 * tt)
    z = analytic_signal(x)
    hv, Lh = _smoothing_window(15, "freq_window")
    gv, Lg = _smoothing_window(7, "time_window")
    W_ref = np.zeros((n, n))
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, Lh)
        taus = np.arange(-taumax, taumax + 1)
        acc = np.empty(taus.size, dtype=complex)
        for a, tau in enumerate(taus):
            mmax = min(Lg, ti - abs(tau), n - 1 - ti - abs(tau))
            ms = np.arange(-mmax, mmax + 1)
            gw = gv[Lg + ms]
            acc[a] = (np.sum(gw * z[ti + tau + ms]
                             * np.conj(z[ti - tau + ms])) / np.sum(gw))
        acc = acc * hv[Lh + taus]
        kernel = np.zeros(n, dtype=complex)
        kernel[(taus + n) % n] = acc
        W_ref[:, ti] = np.real(np.fft.fft(kernel))
    W = wigner_ville(x, fs, freq_window=15, time_window=7)
    np.testing.assert_allclose(W.distribution, W_ref, atol=1e-12)


def test_cwt_rejects_complex_input():
    xc = np.exp(2j * np.pi * 0.1 * np.arange(512))
    with pytest.raises(ConfigurationError, match="complex"):
        cwt(xc, 1000.0)


def test_cwt_explicit_frequency_above_nyquist_raises():
    """An explicit analysis frequency above fs/2 raises a typed error naming
    Nyquist; exactly fs/2 — the top of the default grid — is analysed."""
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(128) / fs)
    with pytest.raises(ConfigurationError, match="Nyquist"):
        cwt(x, fs, frequencies=[50.0, 700.0])
    r = cwt(x, fs, frequencies=[50.0, fs / 2.0])
    assert r.coefficients.shape == (2, 128)
    np.testing.assert_allclose(r.frequencies, [50.0, fs / 2.0])
    assert np.isfinite(r.coefficients).all()


def test_analytic_signal_and_cepstrum_reject_complex_input():
    xc = np.exp(2j * np.pi * 0.05 * np.arange(256))
    with pytest.raises(ConfigurationError, match="complex"):
        analytic_signal(xc)
    with pytest.raises(ConfigurationError, match="complex"):
        cepstrum(xc)


def test_spectrogram_passes_scipy_nperseg_clamp_warning_through():
    """A 512-sample signal against the default nperseg=8192: scipy clamps
    nperseg to the input length, its UserWarning reaches the caller, and the
    result is a single frame."""
    x = np.sin(2 * np.pi * 50 * np.arange(512) / 1000.0)
    with pytest.warns(UserWarning, match="nperseg"):
        r = spectrogram(x, 1000.0)
    assert r.times.size == 1


def test_two_tone_instantaneous_frequency_is_the_mean():
    """Two equal-amplitude tones at f1 and f2: the analytic-signal
    instantaneous frequency reads (f1+f2)/2 — 115 Hz for 100 and 130 Hz, a
    frequency not present in the signal. The trace holds that value wherever
    the beat envelope is away from its nulls; at each null the sampled phase
    steps by ~ -pi, so the naive full-record mean lands near f1 instead."""
    fs, f1, f2 = 2000.0, 100.0, 130.0
    t = np.arange(int(fs)) / fs
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
    fi = instantaneous_frequency(x, fs)
    env = envelope(x)
    core = slice(100, -100)
    assert np.median(fi[core]) == pytest.approx((f1 + f2) / 2, abs=1e-6)
    kept = fi[core][env[core] > 0.25 * env[core].max()]
    assert kept.mean() == pytest.approx((f1 + f2) / 2, abs=1e-3)
    assert np.abs(kept - (f1 + f2) / 2).max() < 1e-6
    assert fi[core].mean() == pytest.approx(f1, abs=2.0)


def test_cwt_and_spectrogram_reject_nonpositive_sample_rate():
    x = np.ones(512)
    with pytest.raises(ConfigurationError, match="sample_rate"):
        cwt(x, 0.0)
    with pytest.raises(ConfigurationError, match="sample_rate"):
        spectrogram(x, 0.0)


def test_inverse_cwt_warns_off_log2_uniform_scale_grid():
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(2048) / fs)
    lin = cwt(x, fs, np.linspace(10, 400, 64))
    with pytest.warns(UserWarning, match="log2"):
        inverse_cwt(lin.coefficients, lin.frequencies, fs)
    single = cwt(x, fs, np.array([50.0]))
    with pytest.warns(UserWarning, match="single scale"):
        inverse_cwt(single.coefficients, single.frequencies, fs)
    # cwt's default log-spaced grid reconstructs silently at the right level.
    log = cwt(x, fs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xr = inverse_cwt(log.coefficients, log.frequencies, fs)
    mid = slice(256, 1792)
    assert abs(np.std(xr[mid]) / np.std(x[mid]) - 1.0) < 0.05


class TestTheCepstrumDelayIsTheDelay:
    def test_a_late_signal_reports_a_positive_delay_and_round_trips(self):
        from uacpy.acoustic_signal.estimate import complex_cepstrum, inverse_complex_cepstrum
        rng = np.random.default_rng(0)
        n = 256
        base = np.zeros(n); base[10:40] = rng.standard_normal(30)
        c0 = complex_cepstrum(base)
        c7 = complex_cepstrum(np.roll(base, 7))
        assert c7.delay - c0.delay == 7
        np.testing.assert_allclose(inverse_complex_cepstrum(c7), np.roll(base, 7), atol=1e-8)
