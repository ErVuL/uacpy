import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch
from uacpy.acoustic_signal.analysis import psd, ppsd, sel, PPSDResult
from uacpy.visualization.plots.signal import (
    plot_psd, plot_ppsd, plot_sel)


def test_psd_matches_welch():
    # These welch arguments are psd()'s effective defaults — window="hann",
    # nperseg=8192, and the noverlap=None that scipy resolves to nperseg//2.
    # Spelling them out makes the default segmentation part of the contract:
    # changing any of them silently changes every caller's spectrum.
    x = np.random.default_rng(0).standard_normal(48000) * 3.0
    f0, p0 = welch(x, 48000.0, window="hann", nperseg=8192, noverlap=4096,
                   scaling="density")
    f, p = psd(x, 48000.0)
    assert np.allclose(f, f0) and np.allclose(p, p0)
    fig, ax = plot_psd(f, p, label="x")
    assert ax.lines
    plt.close(fig)


def test_psd_scaling_density_is_invariant_spectrum_is_not():
    """``scaling='density'`` normalises by the window's noise-equivalent
    bandwidth, so the broadband-noise level is ``2*var/fs`` for every
    nperseg/window combination. ``scaling='spectrum'`` is per-bin power: the
    same noise level carries the NEB ``fs*sum(w**2)/sum(w)**2`` — 4x down
    from nperseg=256 to 1024, 1.5x up from boxcar to hann. An on-bin tone is
    the mirror image: 'spectrum' reads A**2/2 in every combination while
    'density' spreads the tone over the NEB and moves with both knobs."""
    fs = 8000.0
    rng = np.random.default_rng(7)
    noise = 0.1 * rng.standard_normal(16384)
    f0 = 32 * fs / 256.0                    # bin centre for both npersegs
    x = noise + np.cos(2 * np.pi * f0 * np.arange(noise.size) / fs)
    target = 2.0 * np.var(noise) / fs
    dens_noise, spec_noise, spec_tone, dens_tone = {}, {}, {}, {}
    for nper in (256, 1024):
        for win in ("hann", "boxcar"):
            f, pd_ = psd(x, fs, window=win, nperseg=nper, scaling="density")
            _, ps_ = psd(x, fs, window=win, nperseg=nper, scaling="spectrum")
            band = (f > 0) & (f < fs / 2)
            dens_noise[nper, win] = np.median(pd_[band])
            spec_noise[nper, win] = np.median(ps_[band])
            spec_tone[nper, win] = ps_.max()
            dens_tone[nper, win] = pd_.max()
    for level in dens_noise.values():
        assert level == pytest.approx(target, rel=0.06)
    for peak in spec_tone.values():
        assert peak == pytest.approx(0.5, rel=0.02)
    assert (spec_noise[256, "hann"] / spec_noise[1024, "hann"]
            == pytest.approx(4.0, rel=0.1))
    assert (spec_noise[256, "hann"] / spec_noise[256, "boxcar"]
            == pytest.approx(1.5, rel=0.05))
    assert (dens_tone[1024, "hann"] / dens_tone[256, "hann"]
            == pytest.approx(4.0, rel=0.05))
    assert (dens_tone[256, "boxcar"] / dens_tone[256, "hann"]
            == pytest.approx(1.5, rel=0.05))


def test_psd_hann_spectrum_to_density_ratio_is_1p5_bins():
    """At every bin the hann spectrum/density ratio equals the window's
    noise-equivalent bandwidth ``1.5*fs/nperseg`` — the 1.5-bin NEB that makes
    the density estimate window-independent."""
    fs, nper = 8000.0, 256
    x = np.random.default_rng(3).standard_normal(4 * nper)
    f, pd_ = psd(x, fs, nperseg=nper, scaling="density")
    _, ps_ = psd(x, fs, nperseg=nper, scaling="spectrum")
    np.testing.assert_allclose(ps_[1:-1] / pd_[1:-1], 1.5 * fs / nper,
                               rtol=1e-12)


def test_psd_hann_worst_case_scalloping_is_1_42_db():
    """A tone half a bin off centre reads ``8/(3*pi)`` of the on-bin
    amplitude through the default hann window: ``20*log10(8/(3*pi)) =
    -1.4236`` dB, the worst-case 1.42 dB scalloping loss. On a bin centre
    ``scaling='spectrum'`` reads the full ``A**2/2``."""
    fs, nper = 1000.0, 256
    t = np.arange(4 * nper) / fs
    f_on = 32 * fs / nper
    f_off = 32.5 * fs / nper
    _, p_on = psd(np.cos(2 * np.pi * f_on * t), fs, nperseg=nper,
                  scaling="spectrum")
    _, p_off = psd(np.cos(2 * np.pi * f_off * t), fs, nperseg=nper,
                   scaling="spectrum")
    assert p_on.max() == pytest.approx(0.5, rel=1e-6)
    loss_db = 10 * np.log10(p_off.max() / p_on.max())
    assert loss_db == pytest.approx(20 * np.log10(8 / (3 * np.pi)), abs=1e-3)
    assert loss_db == pytest.approx(-1.42, abs=0.01)


def test_ppsd_function():
    x = np.random.default_rng(0).standard_normal(48000 * 4)
    r = ppsd(x, 48000.0, seg_duration=1.0)
    assert isinstance(r, PPSDResult)
    assert r.pdf.shape[1] == r.frequencies.size
    fig, ax = plot_ppsd(r)
    assert ax.collections
    plt.close(fig)


def test_sel_parseval():
    fs, T = 48000.0, 5.0
    t = np.arange(int(T * fs)) / fs
    x = 2.5 * np.sin(2 * np.pi * 1000.0 * t)
    s, bands = sel(x, fs)
    assert abs(np.sum(s) / (np.sum(x ** 2) / fs) - 1.0) < 0.01
    fig, ax = plot_sel(s, bands, duration=T)
    assert ax.patches
    plt.close(fig)


def test_sel_conserves_energy_for_any_nfft():
    """``sel``'s contract is Parseval: the summed band exposure equals
    ``sum(p**2)/fs`` in Pa^2 s. Bin width and segment duration cancel
    (``(fs/nfft) * (nfft/fs) == 1``), so it must hold for any ``nfft``."""
    from scipy.signal import butter, sosfiltfilt
    fs = 48000.0
    rng = np.random.default_rng(0)
    sos = butter(6, [50 / (fs / 2), 15000 / (fs / 2)], btype='band', output='sos')
    x = sosfiltfilt(sos, rng.standard_normal(int(4 * fs)))
    truth = float(np.sum(x ** 2) / fs)
    for nfft in (None, 24000, 96000):
        assert sel(x, fs, nfft=nfft).sel_pa2s.sum() == pytest.approx(truth,
                                                                     rel=1e-4)


def test_sel_third_octave_geometry_is_exact():
    fs = 48000.0
    bands = sel(np.zeros(int(fs)), fs).bands
    centres = np.array([b[1] for b in bands])
    assert np.allclose(centres[1:] / centres[:-1], 2 ** (1 / 3))
    for low, centre, high in bands[:-1]:          # last is clipped to fmax
        assert high / centre == pytest.approx(2 ** (1 / 6))
        assert centre / low == pytest.approx(2 ** (1 / 6))


def test_ppsd_columns_are_densities_with_blank_bins_as_nan():
    """Each frequency column integrates to 1 over the level axis, and bins that
    were never observed are NaN so they plot blank — which is why the result
    must be reduced with nan-aware functions."""
    fs = 8000.0
    rng = np.random.default_rng(1)
    r = ppsd(rng.standard_normal(int(30 * fs)) * 1e-3, fs, seg_duration=1.0,
             nperseg=1024, noverlap=512)
    integral = np.nansum(r.pdf, axis=0) * r.binwidth_db
    assert np.allclose(integral, 1.0)
    assert np.isnan(r.pdf).any() and not np.any(r.pdf == 0)

    centres = (r.level_edges[:-1] + r.level_edges[1:]) / 2
    first_moment = np.nansum(r.pdf * centres[:, None], axis=0) * r.binwidth_db
    band = (r.frequencies > 200) & (r.frequencies < 3500)
    assert np.abs(r.mean_db[band] - first_moment[band]).max() < r.binwidth_db


class TestSELBandGridIsAnchoredAt1kHz:
    """IEC 61260-1 anchors both band systems at 1 kHz — Pierce: "1, 10, 100,
    1000, 10,000 Hz … are also standard 1/3-octave-band f_o's". Snapping the
    ladder to the caller's ``fmin`` instead made the grid move with the
    request: ``fmin=8.9125`` and ``fmin=10.0`` produced disjoint, interleaved
    centres, and the nearest centre to 1 kHz was 1024 Hz (+2.4 %) or 912.3 Hz
    (-8.8 %) depending on it. ``decidecade_bands`` in the same package already
    anchors correctly; ``sel`` was the outlier."""

    FS = 48000

    def _centres(self, **kw):
        y = np.zeros(self.FS)
        return np.array([b[1] for b in sel(y, self.FS, **kw).bands])

    @pytest.mark.parametrize('band_type', ['third_octave', 'octave'])
    def test_one_kilohertz_is_a_band_centre(self, band_type):
        assert np.isclose(self._centres(band_type=band_type), 1000.0).any()

    def test_octave_ladder_is_not_powers_of_two(self):
        # 1024 Hz was reported where a soundscape table expects 1000.
        oc = self._centres(band_type='octave')
        assert not np.isclose(oc, 1024.0).any()

    @pytest.mark.parametrize('fmin', [10.0, 12.0, 20.0, 25.0])
    def test_grids_nest_instead_of_interleaving(self, fmin):
        # The discriminating property: changing fmin may drop bands off the
        # bottom but must never shift the ladder. Before the fix these sets
        # were disjoint from the default one.
        base = self._centres()
        got = self._centres(fmin=fmin)
        assert np.isclose(got[:, None], base[None, :], rtol=1e-9).any(axis=1).all()

    @pytest.mark.parametrize('fs', [2000, 8000, 48000])
    def test_highest_band_stays_below_nyquist(self, fs):
        y = np.zeros(fs)
        highs = np.array([b[2] for b in sel(y, fs).bands])
        assert highs.max() <= fs / 2


class TestPpsdNoverlap:
    """A caller-provided Welch noverlap is respected; it is only replaced
    (with a warning) when the segment clamp leaves no room for it."""

    FS = 8000

    def _sig(self, seconds=4):
        rng = np.random.default_rng(0)
        return rng.standard_normal(self.FS * seconds)

    def test_explicit_noverlap_changes_the_estimate(self):
        x = self._sig()
        a = ppsd(x, self.FS, seg_duration=1.0, nperseg=2048, noverlap=0)
        b = ppsd(x, self.FS, seg_duration=1.0, nperseg=2048, noverlap=1536)
        assert not np.array_equal(np.nan_to_num(a.pdf), np.nan_to_num(b.pdf))

    def test_unfittable_noverlap_warns_and_falls_back(self):
        x = self._sig()
        with pytest.warns(UserWarning, match="noverlap"):
            r = ppsd(x, self.FS, seg_duration=0.1, noverlap=4096)
        assert np.isfinite(np.nansum(r.pdf))

    def test_default_matches_half_nperseg(self):
        x = self._sig()
        d = ppsd(x, self.FS, seg_duration=1.0, nperseg=2048)
        e = ppsd(x, self.FS, seg_duration=1.0, nperseg=2048, noverlap=1024)
        np.testing.assert_array_equal(np.nan_to_num(d.pdf), np.nan_to_num(e.pdf))


def test_sel_warns_when_chunk_size_misaligned_with_nfft():
    fs = 8000
    x = np.sin(2 * np.pi * 1000.5 * np.arange(fs * 4) / fs)
    with pytest.warns(UserWarning, match="multiple of nfft"):
        sel(x, fs, chunk_size=fs + fs // 3)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        sel(x, fs, chunk_size=2 * fs)          # aligned: silent


def test_sel_rejects_complex_input():
    from uacpy.core.exceptions import ConfigurationError
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    with pytest.raises(ConfigurationError, match="complex"):
        sel(np.exp(2j * np.pi * 1000.0 * t), fs)


def test_sel_counts_tone_exactly_on_the_top_band_edge():
    """A bin exactly on edges[-1] belongs to the last band (the top edge is
    closed), so a tone at fmax carries its full exposure."""
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    x = np.sqrt(2.0) * np.cos(2 * np.pi * 3000.0 * t)   # 1 Pa^2 over 1 s
    out = sel(x, fs, band_type="linear", fmin=100, fmax=3000, num_bands=8)
    assert out.sel_pa2s.sum() == pytest.approx(1.0, rel=1e-9)
    assert out.sel_pa2s[-1] == pytest.approx(1.0, rel=1e-9)


def test_sel_rejects_nonpositive_sample_rate():
    from uacpy.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="sample_rate"):
        sel(np.ones(100), 0.0)
