"""Constant-Q transform family (Brown 1991, JASA 89:425).

Validates the theory (constant Q, geometric spacing, per-bin window length) and
the behaviour (a tone peaks in the bin nearest its frequency) for the transform,
PSD, spectrogram, and probabilistic constant-Q, plus the plotters.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from uacpy.acoustic_signal import (  # noqa: E402
    constant_q_transform, constant_q_psd, constant_q_spectrogram,
    probabilistic_constant_q, CQTResult, CQPSDResult, CQSpectrogramResult,
    CQPPSDResult)
from uacpy.acoustic_signal.constant_q import _cq_quality, _cq_frequencies
from uacpy.visualization import (  # noqa: E402
    plot_constant_q_spectrogram, plot_constant_q_psd, plot_constant_q_ppsd)
from uacpy.core.exceptions import ConfigurationError  # noqa: E402

FS = 8000.0


def _tone(freq, dur=2.0, fs=FS):
    return np.sin(2 * np.pi * freq * np.arange(int(dur * fs)) / fs)


# ── theory ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("B", [12, 24, 36])
def test_quality_factor_formula(B):
    # Constant quality factor Q = 1 / (2**(1/B) - 1)  (Brown 1991).
    assert _cq_quality(B) == pytest.approx(1.0 / (2.0 ** (1.0 / B) - 1.0))


@pytest.mark.parametrize("B", [12, 24])
def test_geometric_frequency_spacing(B):
    f = _cq_frequencies(50.0, 2000.0, B)
    assert f[0] == pytest.approx(50.0)
    # consecutive ratio is exactly 2**(1/B)
    np.testing.assert_allclose(f[1:] / f[:-1], 2.0 ** (1.0 / B), rtol=1e-12)
    # an integer number of bins per octave
    octave = f[f <= 100.0]
    assert len(octave) == B + 1 or len(octave) == B  # 50->100 spans one octave


# ── behaviour: a tone peaks in the nearest bin ───────────────────────────────
def test_transform_peaks_at_tone():
    f0 = 440.0
    r = constant_q_transform(_tone(f0), FS, fmin=100, fmax=2000,
                             bins_per_octave=24)
    assert isinstance(r, CQTResult)
    peak = r.frequencies[np.argmax(np.abs(r.coefficients))]
    # within one constant-Q bin of the tone
    assert abs(peak - f0) < f0 * (2.0 ** (1.0 / 24) - 1.0) * 1.5


def test_psd_peaks_at_tone_and_shape():
    f0 = 440.0
    p = constant_q_psd(_tone(f0), FS, fmin=100, fmax=2000, bins_per_octave=24)
    assert isinstance(p, CQPSDResult)
    assert p.power.shape == p.frequencies.shape
    assert np.all(p.power[np.isfinite(p.power)] >= 0)
    peak = p.frequencies[np.nanargmax(p.power)]
    assert abs(peak - f0) < f0 * (2.0 ** (1.0 / 24) - 1.0) * 1.5


def test_spectrogram_shape_and_finite():
    sg = constant_q_spectrogram(_tone(440.0), FS, fmin=100, fmax=2000,
                                bins_per_octave=12)
    assert isinstance(sg, CQSpectrogramResult)
    assert sg.power.shape == (sg.frequencies.size, sg.times.size)
    assert np.isfinite(sg.power).all()
    assert np.all(sg.power >= 0)
    # the tone's bin carries most of the energy
    peak = sg.frequencies[np.argmax(sg.power.mean(axis=1))]
    assert abs(peak - 440.0) < 440.0 * (2.0 ** (1.0 / 12) - 1.0) * 1.5


# ── probabilistic constant-Q ─────────────────────────────────────────────────
class TestConstantQPPSDCarriesTheReferenceItsLevelsAreStatedAgainst:
    """``probabilistic_constant_q`` takes ``ref`` and reports every level as
    dB re ``ref**2``, but the result used not to carry that value, so every
    consumer had to guess it and the plotter hardcoded the package default —
    a 120 dB error for anyone working in µPa. Same fix, and same reasoning, as
    ``ppsd`` / ``PPSDResult``.
    """

    def _run(self, **kw):
        return probabilistic_constant_q(_tone(440.0), FS, fmin=100, fmax=2000,
                                        bins_per_octave=12, ddB=1.0, **kw)

    def test_the_default_reference_is_reported(self):
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER
        assert self._run().ref == REFERENCE_PRESSURE_WATER

    @pytest.mark.parametrize('ref', [1.0, 1e-5, 20e-6])
    def test_a_non_default_reference_is_reported(self, ref):
        assert self._run(ref=ref).ref == ref

    def test_the_reference_tracks_a_real_120_db_move_in_the_levels(self):
        default = self._run()
        pascals = self._run(ref=1.0)
        shift = np.nanmean(pascals.mean_db - default.mean_db)
        assert shift == pytest.approx(-120.0, abs=1e-9)
        assert default.ref != pascals.ref

    def test_the_existing_fields_keep_their_positions(self):
        """Fields carrying what the levels MEAN are appended, never inserted.

        ``ref`` and ``scaling`` both came later, and both went on the end with
        defaults, so a result another suite builds by hand from the six
        original fields still constructs and reports both defaults. The
        assertion is on the PREFIX rather than the whole list: freezing the
        full tuple would fail the next honest append while catching nothing a
        prefix check misses, since a reorder moves one of the six.
        """
        r = self._run()
        original_six = ('frequencies', 'level_edges', 'pdf', 'mean_db',
                        'std_db', 'binwidth_db')
        assert r._fields[:len(original_six)] == original_six
        assert set(r._fields) >= {'ref', 'scaling'}
        built = CQPPSDResult(r.frequencies, r.level_edges, r.pdf, r.mean_db,
                             r.std_db, r.binwidth_db)
        assert built.ref == r.ref
        assert built.scaling == 'spectrum'


def test_probabilistic_constant_q():
    pp = probabilistic_constant_q(_tone(440.0), FS, fmin=100, fmax=2000,
                                  bins_per_octave=12, ddB=1.0)
    assert isinstance(pp, CQPPSDResult)
    assert pp.pdf.shape == (pp.level_edges.size - 1, pp.frequencies.size)
    assert pp.mean_db.shape == pp.frequencies.shape
    # each frequency column integrates to ~1 over the level axis (density)
    col = pp.pdf[:, np.nanargmax(pp.mean_db)]
    integral = np.nansum(col) * pp.binwidth_db
    assert integral == pytest.approx(1.0, abs=0.05)
    # mean level peaks near the tone bin
    assert abs(pp.frequencies[np.nanargmax(pp.mean_db)] - 440.0) < 60.0


# ── validation / robustness ──────────────────────────────────────────────────
def test_validation_errors():
    x = _tone(440.0)
    with pytest.raises(ConfigurationError):
        constant_q_transform(x, FS, fmin=500, fmax=100)          # fmin >= fmax
    with pytest.raises(ConfigurationError):
        constant_q_transform(x, FS, fmin=100, fmax=FS)           # > Nyquist
    with pytest.raises(ConfigurationError):
        constant_q_transform(np.zeros((4, 4)), FS)               # not 1-D
    with pytest.raises(ConfigurationError):
        constant_q_transform(_tone(440.0).astype(complex), FS)   # complex input


def test_short_signal_warns_and_drops_low_bins():
    # fmin so low that the lowest bin's window exceeds the signal length: it
    # never fits a full window, so it is warned about and averaged to NaN.
    short = _tone(440.0, dur=0.02)
    with pytest.warns(UserWarning):
        p = constant_q_psd(short, FS, fmin=20, fmax=2000, bins_per_octave=12)
    assert np.isnan(p.power[0])          # lowest bin: no fully-inside frame
    assert np.isfinite(p.power[-1])      # highest bin: short window fits


# ── scaling: spectrum vs density ─────────────────────────────────────────────
def test_density_scaling_matches_welch_white_noise():
    from scipy.signal import welch
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40000)       # white noise, variance ~ 1
    cq = constant_q_psd(x, FS, fmin=200, fmax=3000, bins_per_octave=12,
                        scaling="density")
    cq_level = float(np.nanmedian(cq.power))
    # one-sided white PSD level (scipy welch density, and analytic 2*var/fs)
    fw, pw = welch(x, FS, nperseg=2048, scaling="density")
    welch_level = float(np.median(pw[(fw > 200) & (fw < 3000)]))
    # Order-of-magnitude bounds only: on this seed the constant-Q median sits
    # 0.2 % from the Welch median and 1 % from the analytic level, so the
    # factor-2 window and rel=0.5 are ~50x looser than the observed spread.
    # They catch a scaling blunder (a missing 2, a per-bin bandwidth) and
    # nothing finer.
    assert 0.5 * welch_level < cq_level < 2.0 * welch_level
    assert cq_level == pytest.approx(2.0 * np.var(x) / FS, rel=0.5)


def test_spectrum_scaling_tone_power():
    # 'spectrum' returns one-sided band power: a unit-amplitude tone peaks at
    # A**2/2 = 0.5 (matches scipy welch scaling='spectrum'), not A**2/4.
    fb = _cq_frequencies(80.0, 4000.0, 24)
    f0 = float(fb[np.argmin(np.abs(fb - 1000.0))])     # tone exactly on a bin
    x = np.sin(2 * np.pi * f0 * np.arange(int(2.0 * FS)) / FS)
    p = constant_q_psd(x, FS, fmin=80, fmax=4000, bins_per_octave=24,
                       scaling="spectrum")
    assert np.nanmax(p.power) == pytest.approx(0.5, rel=0.1)


def test_nan_input_rejected():
    x = _tone(440.0)
    x[5] = np.nan
    with pytest.raises(ConfigurationError):
        constant_q_psd(x, FS, fmin=200, fmax=2000)


def test_spectrum_and_density_differ():
    x = _tone(440.0)
    sp = constant_q_psd(x, FS, fmin=100, fmax=2000, bins_per_octave=12,
                        scaling="spectrum")
    de = constant_q_psd(x, FS, fmin=100, fmax=2000, bins_per_octave=12,
                        scaling="density")
    assert not np.allclose(sp.power, de.power)


def test_default_scaling_is_spectrum_where_linear_psd_is_density():
    """The constant-Q family defaults to scaling='spectrum' (Pa² per bin)
    where the linear estimators default to 'density' (Pa²/Hz): pinned by
    signature and by the default output matching the explicit spelling."""
    import inspect
    from uacpy.acoustic_signal import psd, spectrogram
    for fn in (constant_q_psd, constant_q_spectrogram,
               probabilistic_constant_q):
        assert inspect.signature(fn).parameters["scaling"].default == "spectrum"
    for fn in (psd, spectrogram):
        assert inspect.signature(fn).parameters["scaling"].default == "density"
    x = _tone(440.0, dur=1.0)
    d = constant_q_psd(x, FS, fmin=200, fmax=2000, bins_per_octave=12)
    s = constant_q_psd(x, FS, fmin=200, fmax=2000, bins_per_octave=12,
                       scaling="spectrum")
    np.testing.assert_array_equal(np.nan_to_num(d.power),
                                  np.nan_to_num(s.power))
    de = constant_q_psd(x, FS, fmin=200, fmax=2000, bins_per_octave=12,
                        scaling="density")
    assert not np.allclose(np.nan_to_num(d.power), np.nan_to_num(de.power))
    _, p_default = psd(x, FS, nperseg=1024)
    _, p_density = psd(x, FS, nperseg=1024, scaling="density")
    np.testing.assert_array_equal(p_default, p_density)


def test_scaling_validation():
    with pytest.raises(ConfigurationError):
        constant_q_psd(_tone(440.0), FS, scaling="bogus")
    with pytest.raises(ConfigurationError):
        constant_q_spectrogram(_tone(440.0), FS, scaling="bogus")


def test_edge_exclusion_reduces_padding_bias():
    # Zero-padded edge frames carry less signal and bias the average DOWN.
    # Excluding them (constant_q_psd) raises the estimate at the long-window
    # bins above the naive mean of all spectrogram frames (which keeps the
    # diluting edge frames).
    x = _tone(150.0, dur=2.0)
    fmin, fmax, B = 120, 1000, 12
    psd = constant_q_psd(x, FS, fmin=fmin, fmax=fmax, bins_per_octave=B)
    sg = constant_q_spectrogram(x, FS, fmin=fmin, fmax=fmax, bins_per_octave=B)
    naive = np.nanmean(sg.power, axis=1)        # includes zero-padded edges
    k = int(np.nanargmax(psd.power))            # the tone's bin (longest window)
    assert psd.power[k] >= naive[k]


# ── plotters ─────────────────────────────────────────────────────────────────
def test_plotters_smoke():
    x = _tone(440.0, dur=1.0)
    sg = constant_q_spectrogram(x, FS, fmin=100, fmax=2000, bins_per_octave=12)
    p = constant_q_psd(x, FS, fmin=100, fmax=2000, bins_per_octave=12)
    pp = probabilistic_constant_q(x, FS, fmin=100, fmax=2000, bins_per_octave=12)
    for fig, ax in (plot_constant_q_spectrogram(sg.frequencies, sg.times, sg.power),
                    plot_constant_q_psd(p.frequencies, p.power),
                    plot_constant_q_ppsd(pp)):
        assert fig is not None and ax is not None
    plt.close("all")


def test_plotter_unit_label_switches_with_scaling():
    x = _tone(440.0, dur=1.0)
    p = constant_q_psd(x, FS, fmin=100, fmax=2000, bins_per_octave=12,
                       scaling="density")
    _, ax = plot_constant_q_psd(p.frequencies, p.power, scaling="density")
    assert "Pa²/Hz" in ax.get_ylabel()
    plt.close("all")
    _, ax = plot_constant_q_psd(p.frequencies, p.power, scaling="spectrum")
    lbl = ax.get_ylabel()
    assert "Pa²" in lbl and "/Hz" not in lbl
    plt.close("all")


def test_spectrum_calibration_is_exact_on_a_bin_centre():
    """The 'spectrum' scaling promises a tone of amplitude A peaks at A**2/2.
    That holds on a bin centre; between centres the filterbank scallops, by at
    most the ~1.4 dB the module docstring quotes."""
    from uacpy.acoustic_signal.constant_q import _cq_frequencies
    B, fs, A, fmin = 24, 48000.0, 1.7, 100.0
    f = _cq_frequencies(fmin, 4000.0, B)
    k = int(np.argmin(np.abs(f - 500.0)))
    t = np.arange(int(fs)) / fs

    on = A * np.cos(2 * np.pi * f[k] * t)
    freqs, X = constant_q_transform(on, fs, fmin=fmin, fmax=4000.0,
                                    bins_per_octave=B)
    assert abs(X[k]) == pytest.approx(A / 2, rel=1e-3)
    _, power = constant_q_psd(on, fs, fmin=fmin, fmax=4000.0,
                              bins_per_octave=B, scaling='spectrum')
    assert power.max() == pytest.approx(A ** 2 / 2, rel=1e-3)

    mid = float(np.sqrt(f[k] * f[k + 1]))            # midway between centres
    off = A * np.cos(2 * np.pi * mid * t)
    _, pm = constant_q_psd(off, fs, fmin=fmin, fmax=4000.0, bins_per_octave=B,
                           scaling='spectrum')
    assert 0.0 < -10 * np.log10(pm.max() / (A ** 2 / 2)) < 1.5


def test_kernel_analyses_at_bin_centre_up_to_nyquist():
    """Every bin correlates at exactly f_k, so a tone on ANY bin centre —
    including the short-window bins near Nyquist, where the ceil in
    N_k = ceil(Q*fs/f_k) quantises hardest — reads its A**2/2 band power to
    within a few hundredths of a dB."""
    fs = 8000.0
    f = _cq_frequencies(20.0, fs / 2, 24)
    t = np.arange(int(4 * fs)) / fs
    # skip bins above 0.48*fs: there the short window's mainlobe spans the
    # tone's negative-frequency image and the one-sided power reads high for
    # a real tone regardless of the analysis frequency.
    top = f[-40:]
    for f0 in top[top < 0.48 * fs]:                   # top ~1.7 octaves
        x = np.cos(2 * np.pi * f0 * t)
        r = constant_q_psd(x, fs, fmin=20.0, bins_per_octave=24)
        k = int(np.argmin(np.abs(r.frequencies - f0)))
        err_db = 10 * np.log10(r.power[k] / 0.5)
        assert abs(err_db) < 0.05, f"{err_db:.3f} dB at f0={f0:.1f} Hz"


@pytest.mark.parametrize("B,expected_db", [(6, 1.20), (12, 1.31), (24, 1.37),
                                           (48, 1.39)])
def test_scalloping_loss_matches_the_documented_figure(B, expected_db):
    """Worst-case scalloping is ~1.4 dB, not the ~1.3 dB once documented.

    A tone midway (geometrically) between two centres is read low by both
    neighbouring bins; the deficit of the better of the two is the scallop
    loss. It grows slowly with ``bins_per_octave`` towards the Hann window's
    1.42 dB, because narrower bins put the midpoint further out on a mainlobe
    whose shape Q holds fixed.
    """
    fs, fk = 2000.0, 100.0
    Q = _cq_quality(B)
    losses = []
    for centre, offset in ((fk, 0.5), (fk * 2.0 ** (1.0 / B), -0.5)):
        Nk = max(1, int(np.ceil(Q * fs / centre)))
        n = np.arange(Nk)
        w = np.hanning(Nk + 1)[:-1]                  # get_window('hann', fftbins)
        kernel = (w * np.exp(-2j * np.pi * centre * n / fs)) / w.sum()
        tone = np.cos(2 * np.pi * centre * 2.0 ** (offset / B) * n / fs)
        losses.append(2 * abs(np.sum(tone * kernel)) ** 2)
    loss_db = -10 * np.log10(max(losses) / 0.5)
    assert loss_db == pytest.approx(expected_db, abs=0.02)
    assert loss_db < 1.42


# ── near-Nyquist image leak ──────────────────────────────────────────────────
class TestNearNyquistBinsReadAToneHigh:
    """A real tone at ``f_k`` carries a ``-f_k`` component that the kernel
    demodulates to ``-2 f_k``; as ``f_k`` approaches ``fs/2`` the window stops
    rejecting it and the one-sided band power reads ``1 + |W(2f_k)/sum(w)|**2``
    times the tone's mean-square power.

    The bias rises smoothly through the region rather than switching on at a
    single frequency, so both sides of the 0.01 dB warning threshold are
    checked against the measured curve.
    """

    B = 24

    @staticmethod
    def _one_bin_power(u, fs=FS, amp=1.0, dur=8.0):
        """Band power a single constant-Q bin at ``f_k = u*fs`` reads for a
        cosine of amplitude ``amp`` sitting exactly on it. Truth: ``amp**2/2``."""
        fk = u * fs
        x = amp * np.cos(2 * np.pi * fk * np.arange(int(dur * fs)) / fs)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _, power = constant_q_psd(x, fs, fmin=fk / 1.0000001,
                                      fmax=fk * 1.0000001,
                                      bins_per_octave=TestNearNyquistBinsReadAToneHigh.B)
        return float(power[0])

    @staticmethod
    def _warns(u, fs=FS):
        import warnings as _w
        fk = u * fs
        x = np.cos(2 * np.pi * fk * np.arange(int(2.0 * fs)) / fs)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            constant_q_psd(x, fs, fmin=fk / 1.0000001, fmax=fk * 1.0000001,
                           bins_per_octave=TestNearNyquistBinsReadAToneHigh.B)
        return [str(c.message) for c in caught
                if 'negative-frequency image' in str(c.message)]

    @pytest.mark.parametrize("u, expected_db", [
        (0.4000, 0.0003), (0.4600, 0.0000), (0.4835, 0.0031), (0.4860, 0.0000),
        (0.4875, 0.0128), (0.4920, 0.6791), (0.4935, 1.2130), (0.4990, 2.9525),
    ])
    def test_the_over_read_follows_the_measured_bias_curve(self, u, expected_db):
        got_db = 10 * np.log10(self._one_bin_power(u) / 0.5)
        assert got_db == pytest.approx(expected_db, abs=2e-3)

    def test_the_curve_has_a_null_inside_the_rising_region(self):
        """``u = Q/(2(Q+1)) = 0.48577`` is a null of the bias, not its onset:
        the bias at 0.4835, below it, is larger than the bias at 0.4860."""
        below = 10 * np.log10(self._one_bin_power(0.4835) / 0.5)
        at_null = 10 * np.log10(self._one_bin_power(0.4860) / 0.5)
        assert at_null < below
        assert at_null < 1e-4

    def test_the_warning_threshold_is_crossed_between_0_4872_and_0_4874(self):
        assert self._warns(0.4872) == []
        assert len(self._warns(0.4874)) == 1

    def test_a_bin_well_below_the_region_does_not_warn(self):
        assert self._warns(0.30) == []

    @pytest.mark.parametrize("fs", [2000.0, 8000.0, 32000.0])
    def test_the_bias_tracks_f_over_fs_and_not_the_sample_rate(self, fs):
        got_db = 10 * np.log10(self._one_bin_power(0.4935, fs=fs) / 0.5)
        assert got_db == pytest.approx(1.2130, abs=5e-3)

    @pytest.mark.parametrize("amp", [1e-3, 1.0, 1e3])
    def test_the_bias_is_the_same_fraction_at_every_amplitude(self, amp):
        got_db = 10 * np.log10(
            self._one_bin_power(0.4935, amp=amp) / (0.5 * amp ** 2))
        assert got_db == pytest.approx(1.2130, abs=5e-3)

    def test_broadband_noise_in_the_same_bin_is_unbiased(self):
        """The reason the bias is reported rather than divided out: white
        noise reads ``sigma**2 sum(w**2)/sum(w)**2`` at every bin, so a
        correction sized for a tone would push the noise case off."""
        import warnings as _w
        rng = np.random.default_rng(0)
        x = rng.standard_normal(int(60 * FS))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            freqs, power = constant_q_psd(x, FS, scaling="density")
        db = 10 * np.log10(power / (2.0 / FS))
        assert abs(db[-1]) < 0.4
        assert freqs[-1] / FS > 0.49

    def test_every_estimator_warns_on_the_default_call(self):
        import warnings as _w
        x = np.cos(2 * np.pi * 3948.0 * np.arange(int(2 * FS)) / FS)
        for call in (lambda: constant_q_transform(x, FS),
                     lambda: constant_q_psd(x, FS),
                     lambda: constant_q_spectrogram(x, FS),
                     lambda: probabilistic_constant_q(x, FS)):
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                call()
            assert any('negative-frequency image' in str(c.message)
                       for c in caught)

    @staticmethod
    def _one_bin_power_at_phase(u, phase_deg, fs=FS, dur=8.0):
        """Same as ``_one_bin_power`` but with the tone's phase as an argument
        and the bin placed exactly on ``fmax``, so ``u = 0.5`` is reachable."""
        import warnings as _w
        fk = u * fs
        n = np.arange(int(dur * fs))
        x = np.cos(2 * np.pi * fk * n / fs + np.deg2rad(phase_deg))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _, power = constant_q_psd(
                x, fs, fmin=fk / 1.0000001, fmax=fk,
                bins_per_octave=TestNearNyquistBinsReadAToneHigh.B)
        return float(power[-1])

    @pytest.mark.parametrize("phase_deg", [0.0, 30.0, 45.0, 60.0, 90.0])
    def test_below_nyquist_the_frame_average_makes_the_bias_phase_free(
            self, phase_deg):
        """The 0.005 / 0.68 / 1.21 dB figures are averages over frame phase,
        and below fs/2 the image sits at a non-zero -2 f_k so successive frames
        really do see it at different phases. Any tone phase therefore reads
        the same."""
        got = 10 * np.log10(
            self._one_bin_power_at_phase(0.4990, phase_deg) / 0.5)
        assert got == pytest.approx(2.965, abs=0.02)

    @pytest.mark.parametrize("phase_deg, expected_db", [
        (0.0, 6.0206), (30.0, 4.7712), (45.0, 3.0103), (60.0, 0.0),
    ])
    def test_at_exactly_nyquist_the_reading_follows_the_tone_s_own_phase(
            self, phase_deg, expected_db):
        """At ``f_k = fs/2`` the image lands on DC, where its phase no longer
        turns with the frame, so the frame average that produces the 3.01 dB
        figure does not apply. The reading is ``10*log10(4 cos**2 phi)``: a
        COSINE reads 6.02 dB, twice the 3.01 the module used to state flatly
        for this frequency, and 3.01 is the mean over phase rather than a
        bound."""
        got = 10 * np.log10(
            self._one_bin_power_at_phase(0.5, phase_deg) / 0.5)
        assert got == pytest.approx(expected_db, abs=5e-3), (
            f"a tone at exactly fs/2 with phase {phase_deg:g} deg reads "
            f"{got:+.4f} dB; 10*log10(4 cos**2 phi) is {expected_db:+.4f}")

    def test_a_sine_at_exactly_nyquist_is_identically_zero_on_the_grid(self):
        """The other end of the same phase dependence: sin(pi n) is zero at
        every sample, so the bin reads no power at all rather than 3.01 dB
        of excess."""
        power = self._one_bin_power_at_phase(0.5, 90.0)
        assert power < 1e-20 * 0.5, f"read {power:g}, expected ~0"
