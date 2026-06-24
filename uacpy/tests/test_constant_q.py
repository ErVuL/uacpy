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
