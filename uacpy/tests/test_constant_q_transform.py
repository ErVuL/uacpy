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
    constant_q_transform, constant_q_spectrogram, welch, constant_q,
    probabilistic_constant_q,
    CQTResult, CQSpectrogramResult, SpectralEstimate,
    ProbabilisticSpectralEstimate)
from uacpy.acoustic_signal.cqt import _cq_quality, _cq_frequencies
from uacpy.plot import (  # noqa: E402
    plot_constant_q_transform, plot_constant_q_spectrogram,
    plot_constant_q_psd, plot_constant_q_ppsd)
from uacpy.core.exceptions import ConfigurationError  # noqa: E402
from uacpy.tests.conftest import recorded_warnings


# One function per statistic, so the scaling is the name: these adapters map
# this file's ``scaling=`` parametrisations onto the estimator, and keep its
# subject the constant-Q BEHAVIOUR rather than the call shape. They pin the
# scaling this file was written against ('spectrum'), where the estimator
# defaults to 'density'.
def _cq_spectrum(*args, scaling="spectrum", **kwargs):
    return constant_q(*args, scaling=scaling, **kwargs)


def _cq_probabilistic_spectrum(*args, scaling="spectrum", **kwargs):
    return probabilistic_constant_q(*args, scaling=scaling, **kwargs)


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
    r = constant_q_transform(_tone(f0), FS, freq_min=100, freq_max=2000,
                             bins_per_octave=24)
    assert isinstance(r, CQTResult)
    peak = r.frequencies[np.argmax(np.abs(r.coefficients))]
    # within one constant-Q bin of the tone
    assert abs(peak - f0) < f0 * (2.0 ** (1.0 / 24) - 1.0) * 1.5


def test_psd_peaks_at_tone_and_shape():
    f0 = 440.0
    p = _cq_spectrum(_tone(f0), FS, freq_min=100, freq_max=2000, bins_per_octave=24)
    assert isinstance(p, SpectralEstimate)
    assert p.power.shape == p.frequencies.shape
    assert np.all(p.power[np.isfinite(p.power)] >= 0)
    peak = p.frequencies[np.nanargmax(p.power)]
    assert abs(peak - f0) < f0 * (2.0 ** (1.0 / 24) - 1.0) * 1.5


def test_spectrogram_shape_and_finite():
    sg = constant_q_spectrogram(_tone(440.0), FS, freq_min=100, freq_max=2000,
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
    the Welch histogram carries.
    """

    def _run(self, **kw):
        return _cq_probabilistic_spectrum(_tone(440.0), FS, freq_min=100, freq_max=2000,
                                        bins_per_octave=12, level_step_dB=1.0, **kw)

    def test_the_default_reference_is_reported(self):
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER
        assert self._run().ref == REFERENCE_PRESSURE_WATER

    @pytest.mark.parametrize('ref', [1.0, 1e-5, 20e-6])
    def test_a_non_default_reference_is_reported(self, ref):
        assert self._run(ref=ref).ref == ref

    def test_the_reference_tracks_a_real_120_dB_move_in_the_levels(self):
        default = self._run()
        pascals = self._run(ref=1.0)
        shift = np.nanmean(pascals.mean_dB - default.mean_dB)
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
        assert r._fields == ('frequencies', 'level_edges', 'pdf')
        assert (r.ref, r.scaling, r.method) == (r.ref, 'spectrum',
                                                'constant_q')
        built = ProbabilisticSpectralEstimate(
            r.frequencies, r.level_edges, r.pdf, ref=r.ref,
            scaling='spectrum', method='constant_q')
        assert built.ref == r.ref
        assert built.scaling == 'spectrum'


def test_probabilistic_constant_q():
    pp = _cq_probabilistic_spectrum(_tone(440.0), FS, freq_min=100, freq_max=2000,
                                  bins_per_octave=12, level_step_dB=1.0)
    assert isinstance(pp, ProbabilisticSpectralEstimate)
    assert (pp.method, pp.segment_duration) == ('constant_q', None)
    assert pp.pdf.shape == (pp.level_edges.size - 1, pp.frequencies.size)
    assert pp.mean_dB.shape == pp.frequencies.shape
    # each frequency column integrates to ~1 over the level axis (density)
    col = pp.pdf[:, np.nanargmax(pp.mean_dB)]
    integral = np.nansum(col) * pp.level_step_dB
    assert integral == pytest.approx(1.0, abs=0.05)
    # mean level peaks near the tone bin
    assert abs(pp.frequencies[np.nanargmax(pp.mean_dB)] - 440.0) < 60.0


# ── validation / robustness ──────────────────────────────────────────────────
def test_validation_errors():
    x = _tone(440.0)
    with pytest.raises(ConfigurationError, match='require 0 < freq_min < freq_max'):
        constant_q_transform(x, FS, freq_min=500, freq_max=100)          # freq_min >= freq_max
    with pytest.raises(ConfigurationError,
                       match='is above the Nyquist frequency'):
        constant_q_transform(x, FS, freq_min=100, freq_max=FS)           # > Nyquist
    with pytest.raises(ConfigurationError, match='data must be 1-D'):
        constant_q_transform(np.zeros((4, 4)), FS)               # not 1-D
    with pytest.raises(ConfigurationError, match=r'data must be real \(got complex input\)'):
        constant_q_transform(_tone(440.0).astype(complex), FS)   # complex input


def test_short_signal_warns_and_drops_low_bins():
    # freq_min so low that the lowest bin's window exceeds the signal length: it
    # never fits a full window, so it is warned about and averaged to NaN.
    short = _tone(440.0, dur=0.02)
    with pytest.warns(UserWarning):
        p = _cq_spectrum(short, FS, freq_min=20, freq_max=2000, bins_per_octave=12)
    assert np.isnan(p.power[0])          # lowest bin: no fully-inside frame
    assert np.isfinite(p.power[-1])      # highest bin: short window fits


@pytest.mark.parametrize("estimator,fate,not_fate", [
    (constant_q_transform, "zero-padded window", "dropped"),
    (constant_q_spectrogram, "zero-padded window", "dropped"),
    (_cq_spectrum, "dropped from the average", "zero-padded"),
    (_cq_probabilistic_spectrum, "dropped from the average",
     "zero-padded"),
])
def test_short_signal_warning_names_what_the_estimator_does_with_the_bins(
        estimator, fate, not_fate):
    # The transform and the spectrogram keep every frame, so a bin that never
    # fits a full window is evaluated on a zero-padded one and reads low; the
    # averaging estimators drop it. The warning says which.
    short = _tone(440.0, dur=0.02)
    with pytest.warns(UserWarning, match=fate) as rec:
        estimator(short, FS, freq_min=20, freq_max=2000, bins_per_octave=12)
    short_bin = [str(w.message) for w in rec
                 if "lowest bin needs" in str(w.message)]
    assert len(short_bin) == 1 and not_fate not in short_bin[0]


# ── scaling: spectrum vs density ─────────────────────────────────────────────
def test_density_scaling_matches_welch_white_noise():
    from scipy.signal import welch
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40000)       # white noise, variance ~ 1
    cq = _cq_spectrum(x, FS, freq_min=200, freq_max=3000, bins_per_octave=12,
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
    p = _cq_spectrum(x, FS, freq_min=80, freq_max=4000, bins_per_octave=24,
                       scaling="spectrum")
    assert np.nanmax(p.power) == pytest.approx(0.5, rel=0.1)


def test_nan_input_rejected():
    x = _tone(440.0)
    x[5] = np.nan
    with pytest.raises(ConfigurationError, match='data contains NaN or Inf'):
        _cq_spectrum(x, FS, freq_min=200, freq_max=2000)


def test_spectrum_and_density_differ():
    x = _tone(440.0)
    sp = _cq_spectrum(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12,
                        scaling="spectrum")
    de = _cq_spectrum(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12,
                        scaling="density")
    assert not np.allclose(sp.power, de.power)


def test_each_door_names_the_scaling_it_returns():
    """No estimator takes a ``scaling``: the function IS the statistic, so a
    caller reads which one off the name at the call site. The two spectrogram
    transforms keep theirs, because a spectrogram is a view rather than a
    level."""
    import inspect
    from uacpy.acoustic_signal import spectrogram
    for fn in (welch, constant_q):
        assert "method" not in inspect.signature(fn).parameters
        assert inspect.signature(fn).parameters["scaling"].default == "density"
    assert constant_q(_tone(440.0), FS, freq_min=100, freq_max=2000,
                      scaling="spectrum").scaling == "spectrum"
    assert constant_q(_tone(440.0), FS, freq_min=100,
                      freq_max=2000).scaling == "density"
    assert inspect.signature(
        spectrogram).parameters["scaling"].default == "density"
    assert inspect.signature(
        constant_q_spectrogram).parameters["scaling"].default == "density"
    # The two scalings differ, and differ in the documented way.
    x = _tone(440.0, dur=1.0)
    sp = constant_q(x, FS, freq_min=200, freq_max=2000, bins_per_octave=12,
                    scaling="spectrum")
    de = constant_q(x, FS, freq_min=200, freq_max=2000, bins_per_octave=12)
    assert not np.allclose(np.nan_to_num(sp.power), np.nan_to_num(de.power))


def test_scaling_validation():
    """The estimators cannot be told a scaling at all — Python refuses the
    keyword — while the spectrogram, which still takes one, validates it."""
    with pytest.raises(ConfigurationError, match="scaling"):
        constant_q(_tone(440.0), FS, scaling="bogus")
    with pytest.raises(ConfigurationError, match='scaling must be one of'):
        constant_q_spectrogram(_tone(440.0), FS, scaling="bogus")


def test_edge_exclusion_reduces_padding_bias():
    # Zero-padded edge frames carry less signal and bias the average DOWN.
    # Excluding them (constant_q) raises the estimate at the long-window
    # bins above the naive mean of all spectrogram frames (which keeps the
    # diluting edge frames).
    x = _tone(150.0, dur=2.0)
    freq_min, freq_max, B = 120, 1000, 12
    psd = _cq_spectrum(x, FS, freq_min=freq_min, freq_max=freq_max, bins_per_octave=B)
    sg = constant_q_spectrogram(x, FS, freq_min=freq_min, freq_max=freq_max, bins_per_octave=B,
                                scaling='spectrum')
    naive = np.nanmean(sg.power, axis=1)        # includes zero-padded edges
    k = int(np.nanargmax(psd.power))            # the tone's bin (longest window)
    assert psd.power[k] >= naive[k]


# ── coverage: every sample reaches every bin ─────────────────────────────────
class TestEveryBinReadsEverySample:
    """Each bin steps by a quarter of its own kernel, so a transient shorter
    than the lowest bin's kernel reaches the high bins wherever it falls.

    The case is a 20 ms, 1 Pa, Hann-gated tone burst on a ~10 kHz bin in 2 s
    of 1 mPa white noise at 48 kHz, with the default ``freq_min = 20`` Hz: the
    lowest kernel is 81 906 samples and the 10 kHz one 165. Five onsets span
    more than an eighth of the lowest kernel (213 ms), the step every bin
    shared when the frames sat on one grid, so at least four of them fall
    between frames of that grid.
    """

    fs = 48000.0
    duration = 2.0
    freq_max = 12000.0
    onsets = (0.50, 0.55, 0.60, 0.65, 0.70)

    @classmethod
    def _bin(cls):
        ladder = _cq_frequencies(20.0, cls.freq_max, 24)
        k = int(np.argmin(np.abs(ladder - 10000.0)))
        return k, float(ladder[k])

    @classmethod
    def _record(cls, onset):
        _, fk = cls._bin()
        n = int(cls.duration * cls.fs)
        x = 1e-3 * np.random.default_rng(0).standard_normal(n)
        m = int(0.020 * cls.fs)
        i0 = int(onset * cls.fs)
        t = np.arange(m) / cls.fs
        burst = np.hanning(m) * np.sin(2 * np.pi * fk * t)
        x[i0:i0 + m] += burst
        return x, burst

    @pytest.mark.parametrize("onset", onsets)
    def test_the_average_reads_the_burst_power_wherever_it_falls(self, onset):
        k, _ = self._bin()
        x, burst = self._record(onset)
        # The burst's time-averaged band power over the record: its whole
        # spectrum sits inside the 10 kHz bin (293 Hz wide at 24 per octave).
        want = np.sum(burst ** 2) / x.size
        got = _cq_spectrum(x, self.fs, freq_max=self.freq_max).power[k]
        assert got == pytest.approx(want, rel=0.05)

    @pytest.mark.parametrize("onset", onsets)
    def test_the_spectrogram_carries_the_burst_energy_wherever_it_falls(
            self, onset):
        k, _ = self._bin()
        x, burst = self._record(onset)
        sg = constant_q_spectrogram(x, self.fs, freq_max=self.freq_max,
                                    scaling="spectrum")
        cell = sg.times[1] - sg.times[0]
        # Power times cell duration, summed over the cells, is the energy the
        # bin saw; the background adds 1.7e-8 Pa² x 2 s.
        energy = float(np.sum(sg.power[k]) * cell)
        assert energy == pytest.approx(np.sum(burst ** 2) / self.fs,
                                       rel=0.05)

    @pytest.mark.parametrize("onset", onsets)
    def test_the_histogram_holds_the_burst_levels_wherever_it_falls(
            self, onset):
        k, _ = self._bin()
        x, _ = self._record(onset)
        pp = _cq_probabilistic_spectrum(x, self.fs, freq_max=self.freq_max)
        # The burst peaks at 0.5 Pa² band power, 117 dB re 1 µPa²; the
        # background sits near 43 dB.
        loud = pp.level_edges[:-1] >= 110.0
        assert np.isfinite(pp.pdf[loud, k]).any()

    def test_a_burst_shorter_than_the_kernel_reads_the_same_at_every_offset(
            self):
        # A 2 ms burst inside the 3.4 ms kernel weighs as the squared window
        # summed over the frames that cover it: flat for Hann² stepped by a
        # quarter kernel, rippling by 3 dB at a half-kernel step. Twelve
        # onsets one sample apart and beyond span a whole 41-sample step.
        k, fk = self._bin()
        n = int(1.0 * self.fs)
        m = int(0.002 * self.fs)
        burst = np.hanning(m) * np.sin(2 * np.pi * fk * np.arange(m) / self.fs)
        read = []
        for i0 in 24000 + np.array([0, 1, 3, 5, 8, 11, 15, 20, 26, 31, 36,
                                    40]):
            x = np.zeros(n)
            x[i0:i0 + m] = burst
            read.append(_cq_spectrum(x, self.fs, freq_max=self.freq_max).power[k])
        spread_dB = 10 * np.log10(max(read) / min(read))
        assert spread_dB < 0.05

    def test_a_stationary_on_bin_tone_reads_its_full_power(self):
        k, fk = self._bin()
        n = int(self.duration * self.fs)
        x = np.sin(2 * np.pi * fk * np.arange(n) / self.fs)
        got = _cq_spectrum(x, self.fs, freq_max=self.freq_max).power[k]
        assert got == pytest.approx(0.5, rel=1e-4)


def test_spectrogram_defaults_to_density_like_its_siblings():
    """``constant_q_spectrogram`` defaults to the density ``constant_q`` and
    ``spectrogram`` default to, and carries the scaling it used: the default
    frame average equals the explicit density one, and differs from the
    band-power one by each bin's noise-equivalent bandwidth."""
    x = _tone(440.0, dur=1.0)
    kw = dict(freq_min=100, freq_max=2000, bins_per_octave=12)
    default = constant_q_spectrogram(x, FS, **kw)
    density = constant_q_spectrogram(x, FS, scaling='density', **kw)
    band = constant_q_spectrogram(x, FS, scaling='spectrum', **kw)
    assert default.scaling == 'density' and band.scaling == 'spectrum'
    np.testing.assert_array_equal(default.power, density.power)
    assert not np.allclose(default.power, band.power)


def test_the_result_plot_labels_the_unit_it_carries():
    """``.plot()`` reads ``scaling`` off the result, so a density panel is
    labelled per hertz and a band-power panel is not."""
    x = _tone(440.0, dur=1.0)
    kw = dict(freq_min=100, freq_max=2000, bins_per_octave=12)
    for scaling, per_hz in (('density', True), ('spectrum', False)):
        fig, ax = constant_q_spectrogram(x, FS, scaling=scaling, **kw).plot()
        label = fig.axes[-1].get_ylabel()
        assert ('/Hz' in label) is per_hz, (scaling, label)
        plt.close(fig)


# ── plotters ─────────────────────────────────────────────────────────────────
def test_plotters_smoke():
    x = _tone(440.0, dur=1.0)
    sg = constant_q_spectrogram(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12)
    p = _cq_spectrum(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12)
    pp = _cq_probabilistic_spectrum(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12)
    cqt = constant_q_transform(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12)
    for fig, ax in (plot_constant_q_transform(cqt.frequencies, cqt.coefficients),
                    plot_constant_q_spectrogram(sg.frequencies, sg.times, sg.power),
                    plot_constant_q_psd(p.frequencies, p.power),
                    plot_constant_q_ppsd(pp)):
        assert fig is not None and ax is not None
    plt.close("all")


def test_the_transform_plotter_draws_the_magnitude_on_a_log_axis():
    """The coefficients are complex, so the line is ``|X_cq|``, and the
    frequency axis is geometric like the bins it draws."""
    x = _tone(440.0, dur=1.0)
    cqt = constant_q_transform(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12)
    _, ax = plot_constant_q_transform(cqt.frequencies, cqt.coefficients)
    drawn = ax.lines[0].get_ydata()
    assert ax.get_xscale() == "log"
    np.testing.assert_allclose(drawn, np.abs(cqt.coefficients))
    assert np.isrealobj(drawn)
    plt.close("all")


@pytest.mark.parametrize("n_coefficients, raises", [(31, True), (32, False)])
def test_the_transform_plotter_requires_one_coefficient_per_frequency(
        n_coefficients, raises):
    """Both sides of the length check: a coefficient array one short of the
    frequency axis is a mismatched pair, not a shorter curve."""
    frequencies = np.geomspace(100.0, 2000.0, 32)
    coefficients = np.ones(n_coefficients, dtype=complex)
    if raises:
        with pytest.raises(ConfigurationError, match="one value per frequency"):
            plot_constant_q_transform(frequencies, coefficients)
    else:
        fig, _ = plot_constant_q_transform(frequencies, coefficients)
        assert fig is not None
    plt.close("all")


def test_plotter_unit_label_switches_with_scaling():
    x = _tone(440.0, dur=1.0)
    p = _cq_spectrum(x, FS, freq_min=100, freq_max=2000, bins_per_octave=12,
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
    from uacpy.acoustic_signal.cqt import _cq_frequencies
    B, fs, A, freq_min = 24, 48000.0, 1.7, 100.0
    f = _cq_frequencies(freq_min, 4000.0, B)
    k = int(np.argmin(np.abs(f - 500.0)))
    t = np.arange(int(fs)) / fs

    on = A * np.cos(2 * np.pi * f[k] * t)
    freqs, X = constant_q_transform(on, fs, freq_min=freq_min, freq_max=4000.0,
                                    bins_per_octave=B)
    assert abs(X[k]) == pytest.approx(A / 2, rel=1e-3)
    _, power = _cq_spectrum(on, fs, freq_min=freq_min, freq_max=4000.0,
                              bins_per_octave=B, scaling='spectrum')
    assert power.max() == pytest.approx(A ** 2 / 2, rel=1e-3)

    mid = float(np.sqrt(f[k] * f[k + 1]))            # midway between centres
    off = A * np.cos(2 * np.pi * mid * t)
    _, pm = _cq_spectrum(off, fs, freq_min=freq_min, freq_max=4000.0, bins_per_octave=B,
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
        r = _cq_spectrum(x, fs, freq_min=20.0, bins_per_octave=24)
        k = int(np.argmin(np.abs(r.frequencies - f0)))
        err_dB = 10 * np.log10(r.power[k] / 0.5)
        assert abs(err_dB) < 0.05, f"{err_dB:.3f} dB at f0={f0:.1f} Hz"


@pytest.mark.parametrize("B,expected_dB", [(6, 1.20), (12, 1.31), (24, 1.37),
                                           (48, 1.39)])
def test_scalloping_loss_matches_the_documented_figure(B, expected_dB):
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
    loss_dB = -10 * np.log10(max(losses) / 0.5)
    assert loss_dB == pytest.approx(expected_dB, abs=0.02)
    assert loss_dB < 1.42


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
            _, power = _cq_spectrum(x, fs, freq_min=fk / 1.0000001,
                                      freq_max=fk * 1.0000001,
                                      bins_per_octave=TestNearNyquistBinsReadAToneHigh.B)
        return float(power[0])

    @staticmethod
    def _warns(u, fs=FS):
        fk = u * fs
        x = np.cos(2 * np.pi * fk * np.arange(int(2.0 * fs)) / fs)
        with recorded_warnings() as caught:
            _cq_spectrum(x, fs, freq_min=fk / 1.0000001, freq_max=fk * 1.0000001,
                           bins_per_octave=TestNearNyquistBinsReadAToneHigh.B)
        return [str(c.message) for c in caught
                if 'negative-frequency image' in str(c.message)]

    @pytest.mark.parametrize("u, expected_dB", [
        (0.4000, 0.0003), (0.4600, 0.0000), (0.4835, 0.0031), (0.4860, 0.0000),
        (0.4875, 0.0128), (0.4920, 0.6791), (0.4935, 1.2130), (0.4990, 2.9525),
    ])
    def test_the_over_read_follows_the_measured_bias_curve(self, u, expected_dB):
        got_dB = 10 * np.log10(self._one_bin_power(u) / 0.5)
        assert got_dB == pytest.approx(expected_dB, abs=2e-3)

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
        got_dB = 10 * np.log10(self._one_bin_power(0.4935, fs=fs) / 0.5)
        assert got_dB == pytest.approx(1.2130, abs=5e-3)

    @pytest.mark.parametrize("amp", [1e-3, 1.0, 1e3])
    def test_the_bias_is_the_same_fraction_at_every_amplitude(self, amp):
        got_dB = 10 * np.log10(
            self._one_bin_power(0.4935, amp=amp) / (0.5 * amp ** 2))
        assert got_dB == pytest.approx(1.2130, abs=5e-3)

    def test_broadband_noise_in_the_same_bin_is_unbiased(self):
        """The reason the bias is reported rather than divided out: white
        noise reads ``sigma**2 sum(w**2)/sum(w)**2`` at every bin, so a
        correction sized for a tone would push the noise case off."""
        import warnings as _w
        rng = np.random.default_rng(0)
        x = rng.standard_normal(int(60 * FS))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            freqs, power = _cq_spectrum(x, FS, scaling="density",
                                        freq_max=FS / 2)
        dB = 10 * np.log10(power / (2.0 / FS))
        assert abs(dB[-1]) < 0.4
        assert freqs[-1] / FS > 0.49

    def test_every_estimator_warns_on_an_explicit_nyquist_fmax_only(self):
        """An unset ``freq_max`` stops below the leaking bins, so the default
        call has nothing to warn about; asking for the Nyquist bin by name
        is warned on every estimator."""
        x = np.cos(2 * np.pi * 3948.0 * np.arange(int(2 * FS)) / FS)
        for estimator in (constant_q_transform, _cq_spectrum,
                          constant_q_spectrogram, _cq_probabilistic_spectrum):
            for kwargs, warned in (({}, False), ({'freq_max': FS / 2}, True)):
                with recorded_warnings() as caught:
                    estimator(x, FS, **kwargs)
                assert any('negative-frequency image' in str(c.message)
                           for c in caught) is warned, (estimator, kwargs)

    def test_the_default_range_ends_on_the_last_bin_below_the_threshold(self):
        """Both sides of the cut: the default grid's top bin reads its tone
        without a warning, and the next bin of the same ladder would warn."""
        freqs = _cq_spectrum(np.random.default_rng(0).standard_normal(
            int(2 * FS)), FS, bins_per_octave=self.B).frequencies
        top = freqs[-1] / FS
        above = top * 2.0 ** (1.0 / self.B)
        assert self._warns(top) == []
        assert len(self._warns(above)) == 1

    @staticmethod
    def _one_bin_power_at_phase(u, phase_deg, fs=FS, dur=8.0):
        """Same as ``_one_bin_power`` but with the tone's phase as an argument
        and the bin placed exactly on ``freq_max``, so ``u = 0.5`` is reachable."""
        import warnings as _w
        fk = u * fs
        n = np.arange(int(dur * fs))
        x = np.cos(2 * np.pi * fk * n / fs + np.deg2rad(phase_deg))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _, power = _cq_spectrum(
                x, fs, freq_min=fk / 1.0000001, freq_max=fk,
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

    @pytest.mark.parametrize("phase_deg, expected_dB", [
        (0.0, 6.0206), (30.0, 4.7712), (45.0, 3.0103), (60.0, 0.0),
    ])
    def test_at_exactly_nyquist_the_reading_follows_the_tone_s_own_phase(
            self, phase_deg, expected_dB):
        """At ``f_k = fs/2`` the image lands on DC, where its phase no longer
        turns with the frame, so the frame average that produces the 3.01 dB
        figure does not apply. The reading is ``10*log10(4 cos**2 phi)``: a
        COSINE reads 6.02 dB, twice the 3.01 the module used to state flatly
        for this frequency, and 3.01 is the mean over phase rather than a
        bound."""
        got = 10 * np.log10(
            self._one_bin_power_at_phase(0.5, phase_deg) / 0.5)
        assert got == pytest.approx(expected_dB, abs=5e-3), (
            f"a tone at exactly fs/2 with phase {phase_deg:g} deg reads "
            f"{got:+.4f} dB; 10*log10(4 cos**2 phi) is {expected_dB:+.4f}")

    def test_a_sine_at_exactly_nyquist_is_identically_zero_on_the_grid(self):
        """The other end of the same phase dependence: sin(pi n) is zero at
        every sample, so the bin reads no power at all rather than 3.01 dB
        of excess."""
        power = self._one_bin_power_at_phase(0.5, 90.0)
        assert power < 1e-20 * 0.5, f"read {power:g}, expected ~0"


def test_constant_q_runs_along_the_named_axis_of_a_multichannel_record():
    """``axis=`` means what it means on ``welch``: a ``(n_samples,
    n_channels)`` record at ``axis=0`` gives one spectrum per channel,
    frequency last, each equal to the channel's own 1-D estimate."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((int(FS), 3))
    kw = dict(freq_min=100.0, freq_max=2000.0, bins_per_octave=12)
    est = _cq_spectrum(x, FS, axis=0, **kw)
    assert est.power.shape == (3, est.frequencies.size)
    for ch in range(3):
        np.testing.assert_array_equal(est.power[ch],
                                      _cq_spectrum(x[:, ch], FS, **kw).power)


@pytest.mark.parametrize('axis', [0, 1])
def test_constant_q_channels_come_out_in_welchs_orientation(axis):
    """For the same multichannel input and ``axis``, ``constant_q`` and
    ``welch`` both return one spectrum per channel with frequency LAST, the
    channels in input order, so one caller handles both estimates."""
    from uacpy.acoustic_signal import welch
    rng = np.random.default_rng(2)
    block = rng.standard_normal((int(FS), 2))
    block[:, 1] *= 10.0                       # channel 1 is 20 dB louder
    x = block if axis == 0 else block.T
    kw = dict(freq_min=100.0, freq_max=2000.0, bins_per_octave=12)
    cq = _cq_spectrum(x, FS, axis=axis, **kw)
    w = welch(x, FS, axis=axis, nperseg=1024)
    assert cq.power.shape[:-1] == w.power.shape[:-1] == (2,)
    assert cq.power.shape[-1] == cq.frequencies.size
    assert w.power.shape[-1] == w.frequencies.size
    for est in (cq, w):
        assert np.nanmedian(est.power[1] / est.power[0]) == pytest.approx(
            100.0, rel=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# acoustic_signal/cqt.py — frequency ladder, kernel, frames, hop, ppsd
# ─────────────────────────────────────────────────────────────────────────────


class TestConstantQFrequencyLadder:
    """``f_k = freq_min·2**(k/B)`` with ``K = floor(B·log2(freq_max/freq_min)) + 1``
    bins; freq_min below 1 Hz is legal, freq_min = 0 and freq_max <= freq_min are not."""

    def test_exact_octave_ladder(self):
        from uacpy.acoustic_signal.cqt import constant_q_transform
        x = np.sin(2 * np.pi * 200.0 * np.arange(4096) / 4096.0)
        r = constant_q_transform(x, 4096.0, freq_min=100.0, freq_max=400.0,
                                 bins_per_octave=1)
        np.testing.assert_allclose(r.frequencies, [100.0, 200.0, 400.0],
                                   rtol=1e-12)

    def test_sub_hertz_fmin_is_legal(self):
        from uacpy.acoustic_signal.cqt import _cq_frequencies
        f = _cq_frequencies(0.5, 2.0, 1)
        np.testing.assert_allclose(f, [0.5, 1.0, 2.0], rtol=1e-12)

    def test_fmin_zero_raises(self):
        from uacpy.acoustic_signal.cqt import _cq_frequencies
        with pytest.raises(ConfigurationError, match="0 < freq_min < freq_max"):
            _cq_frequencies(0.0, 100.0, 24)

    def test_fmax_equal_to_fmin_raises(self):
        from uacpy.acoustic_signal.cqt import _cq_frequencies
        with pytest.raises(ConfigurationError, match="0 < freq_min < freq_max"):
            _cq_frequencies(100.0, 100.0, 24)


class TestConstantQKernelConstruction:
    """The kernel is ``w·exp(-2j·pi·f_k·n/fs)/Σw`` with a *periodic*
    (fftbins) window of ``N_k = max(1, ceil(Q·fs/f_k))`` samples — pinned
    against an independent reconstruction, phase included."""

    def test_kernel_matches_definition_exactly(self):
        from scipy.signal import get_window
        from uacpy.acoustic_signal.cqt import _cq_kernels
        fs, fk, Q = 1000.0, 125.0, 16.817
        (Nk, ker, _), = _cq_kernels(np.array([fk]), Q, fs, "hann")
        assert Nk == int(np.ceil(Q * fs / fk))
        w = get_window("hann", Nk, fftbins=True)
        n = np.arange(Nk)
        want = (w * np.exp(-2j * np.pi * fk * n / fs)) / float(np.sum(w))
        np.testing.assert_allclose(ker, want, rtol=0, atol=1e-15)

    def test_window_floor_is_one_sample(self):
        from uacpy.acoustic_signal.cqt import _cq_kernels
        # Q·fs/f_k = 0.5 -> ceil = 1: the floor keeps N_k = 1, not 2.
        (Nk, _, _), = _cq_kernels(np.array([2000.0]), 1.0, 1000.0, "hann")
        assert Nk == 1


class TestConstantQFrameGeometry:
    """Window centring, the exact-fit validity boundary, and the
    zero-padded edge path, pinned with unit impulses."""

    def test_window_is_centred_on_the_requested_sample(self):
        from uacpy.acoustic_signal.cqt import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk, ker, _ = kernels[0]
        x = np.zeros(4 * Nk)
        centre = 2 * Nk
        x[centre] = 1.0
        coeffs, valid = _cq_frame(x, centre, kernels)
        # The impulse sits at window index Nk//2, so the coefficient is
        # that single kernel sample.
        assert valid[0]
        np.testing.assert_allclose(coeffs[0], ker[Nk // 2], rtol=0,
                                   atol=1e-15)

    def test_exact_fit_window_is_valid_one_short_is_not(self):
        from uacpy.acoustic_signal.cqt import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk = kernels[0][0]
        x = np.ones(Nk)
        _, valid_fit = _cq_frame(x, Nk // 2, kernels)
        assert valid_fit[0]
        _, valid_short = _cq_frame(x[:-1], Nk // 2, kernels)
        assert not valid_short[0]

    def test_edge_padding_keeps_sample_zero(self):
        from uacpy.acoustic_signal.cqt import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk, ker, _ = kernels[0]
        x = np.zeros(Nk)
        x[0] = 1.0
        # Centre at 0: the window start is negative, so x[0] lands at
        # kernel index Nk//2 via the zero-padded path.
        coeffs, valid = _cq_frame(x, 0, kernels)
        assert not valid[0]
        np.testing.assert_allclose(coeffs[0], ker[Nk // 2], rtol=0,
                                   atol=1e-15)


class TestConstantQSpectrogramTimeAxis:
    """``times = arange(0, n, hop)/fs`` — starts at zero, in seconds."""

    def test_times_start_at_zero_in_seconds(self):
        from uacpy.acoustic_signal.cqt import constant_q_spectrogram
        fs = 2000.0
        x = np.sin(2 * np.pi * 250.0 * np.arange(2048) / fs)
        r = constant_q_spectrogram(x, fs, freq_min=125.0, freq_max=500.0,
                                   bins_per_octave=2, hop=100)
        np.testing.assert_allclose(
            r.times, np.arange(0, 2048, 100) / fs, rtol=1e-12)


class TestConstantQHopResolution:
    """Default hop is ``max(1, min(n_lowest//8, max(1, n//8)))`` from the
    *lowest* bin's window; an explicit ``hop=1`` is legal."""

    def test_default_hop_follows_the_lowest_bin(self):
        from uacpy.acoustic_signal.cqt import _resolve_hop
        kernels = [(100, None, None), (50, None, None)]
        assert _resolve_hop(None, kernels, 10000, "t") == 100 // 8

    def test_tiny_signal_floors_at_one(self):
        from uacpy.acoustic_signal.cqt import _resolve_hop
        assert _resolve_hop(None, [(8, None, None)], 8, "t") == 1

    def test_explicit_hop_of_one_is_accepted(self):
        from uacpy.acoustic_signal.cqt import _resolve_hop
        assert _resolve_hop(1, [(100, None, None)], 1000, "t") == 1
        with pytest.raises(ConfigurationError, match="hop must be >= 1"):
            _resolve_hop(0, [(100, None, None)], 1000, "t")


class TestConstantQSetupWarningBoundary:
    """The too-short-signal warning keys on the lowest bin needing *more*
    samples than the signal has — an exact fit stays silent."""

    def test_exact_fit_does_not_warn(self):
        from uacpy.acoustic_signal.cqt import (
            _cq_frequencies, _cq_kernels, _cq_quality, _cq_setup,
        )
        fs, freq_min, B = 1000.0, 100.0, 2
        n_lowest = _cq_kernels(
            _cq_frequencies(freq_min, fs / 2, B), _cq_quality(B), fs,
            "hann")[0][0]
        with recorded_warnings() as caught:
            _cq_setup(np.zeros(n_lowest), fs, freq_min, None, B, "hann", "t",
                      drops_short_bins=True)
        # Only this warning is under test. `freq_max=None` resolves to fs/2, so
        # the near-Nyquist image note fires on the same call by design.
        assert not any("lowest bin needs" in str(c.message) for c in caught)
        with pytest.warns(UserWarning, match="lowest bin needs"):
            _cq_setup(np.zeros(n_lowest - 1), fs, freq_min, None, B, "hann", "t",
                      drops_short_bins=True)


class TestConstantQTransformCentresTheFrame:
    """``constant_q_transform`` analyses one frame centred on ``n//2``."""

    def test_impulse_at_the_centre_sample(self):
        from uacpy.acoustic_signal.cqt import (
            _cq_frequencies, _cq_kernels, _cq_quality, constant_q_transform,
        )
        fs, freq_min, freq_max, B = 1000.0, 100.0, 200.0, 1
        n = 1024
        x = np.zeros(n)
        x[n // 2] = 1.0
        r = constant_q_transform(x, fs, freq_min=freq_min, freq_max=freq_max,
                                 bins_per_octave=B)
        kernels = _cq_kernels(_cq_frequencies(freq_min, freq_max, B),
                              _cq_quality(B), fs, "hann")
        for got, (Nk, ker, _) in zip(r.coefficients, kernels):
            np.testing.assert_allclose(got, ker[Nk // 2], rtol=0, atol=1e-15)


class TestProbabilisticConstantQContracts:
    """Level-edge defaults run ``level_min_dB..level_max_dB`` inclusive, and a bin
    with exactly one fully-inside frame is data — for the PSD average and
    the PPSD histogram both — not a NaN column."""

    def _one_frame_case(self):
        from uacpy.acoustic_signal.cqt import (
            _cq_frequencies, _cq_kernels, _cq_quality,
        )
        fs, freq_min, B = 1000.0, 100.0, 2
        n_lowest = _cq_kernels(
            _cq_frequencies(freq_min, fs / 2, B), _cq_quality(B), fs,
            "hann")[0][0]
        # A record exactly as long as the lowest bin's window: that window
        # fits in one place only -> exactly one valid frame for bin 0.
        x = np.sin(2 * np.pi * freq_min * np.arange(n_lowest) / fs)
        return x, fs, freq_min, B, n_lowest

    def test_level_edges_run_from_lvlmin_to_lvlmax_inclusive(self):
        from uacpy.acoustic_signal import probabilistic_constant_q
        x, fs, freq_min, B, n_lowest = self._one_frame_case()
        r = probabilistic_constant_q(x, fs, scaling='spectrum', freq_min=freq_min, bins_per_octave=B,
                                     level_step_dB=1.0, level_min_dB=0, level_max_dB=150)
        assert r.level_edges[0] == 0.0
        assert r.level_edges[-1] == 150.0
        assert r.level_edges.size == 151

    def test_single_valid_frame_is_data(self):
        from uacpy.acoustic_signal import (constant_q,
                                           probabilistic_constant_q)
        x, fs, freq_min, B, n_lowest = self._one_frame_case()
        psd = constant_q(x, fs, scaling='spectrum', freq_min=freq_min,
                             bins_per_octave=B)
        assert np.isfinite(psd.power[0])
        ppsd = probabilistic_constant_q(x, fs, scaling='spectrum', freq_min=freq_min, bins_per_octave=B)
        assert np.isfinite(ppsd.pdf[:, 0]).any()


class TestProbabilisticConstantQDefaultLevels:
    """The *default* level range is 0..150 dB — pinned without passing
    level_min_dB/level_max_dB explicitly."""

    def test_default_edges(self):
        from uacpy.acoustic_signal import probabilistic_constant_q
        x = np.sin(2 * np.pi * 100.0 * np.arange(4096) / 1000.0)
        r = probabilistic_constant_q(x, 1000.0, scaling='spectrum', freq_min=100.0,
                                     bins_per_octave=2)
        assert r.level_edges[0] == 0.0
        assert r.level_edges[-1] == 150.0


class TestConstantQHopInnerFloor:
    """The signal-length term of the default hop, ``max(1, n//8)``, floors
    at one for signals under 16 samples even when the lowest window is
    longer."""

    def test_short_signal_with_long_window(self):
        from uacpy.acoustic_signal.cqt import _resolve_hop
        assert _resolve_hop(None, [(24, None, None)], 8, "t") == 1
