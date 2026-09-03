import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch
from uacpy.acoustic_signal.analysis import psd, ppsd, sel, PPSDResult
import warnings

from uacpy.acoustic_signal.bands import decidecade_band_levels
from uacpy.acoustic_signal.timefreq import spectrogram
from uacpy.core.exceptions import ConfigurationError

#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]
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


class TestBothOctaveLaddersComeFromOneLoop:
    """The octave and third-octave branches were the same loop with a
    different step, and merging them is bit-identical — over 560
    ``(fmin, fmax, sample_rate, band_type)`` combinations, every edge
    reproduced to the last bit — PROVIDED the half-step is written
    ``math.pow(2, step/2)``.

    That form is what makes the equality guaranteed rather than lucky:
    ``(1/3)/2 == 1/6`` exactly in binary floating point (halving is exact), so
    ``math.pow(2, step/2)`` returns the identical double the third-octave
    branch's ``math.pow(2, 1/6)`` did. The tests above compare edges with
    ``pytest.approx``, so they would not notice a rewrite that moved them by an
    ULP; these assert equality.
    """

    FS = 48000.0

    @staticmethod
    def _bands(band_type):
        from uacpy.acoustic_signal.analysis import _sel_bands
        return _sel_bands(8.9125, 22387.0, band_type, 30,
                          TestBothOctaveLaddersComeFromOneLoop.FS)

    def test_the_exponent_identity_the_merge_rests_on_holds(self):
        import math
        assert (1.0 / 3.0) / 2 == 1.0 / 6.0
        assert math.pow(2, (1.0 / 3.0) / 2) == math.pow(2, 1 / 6)
        assert math.pow(2, 1.0 / 2) == math.sqrt(2)

    @pytest.mark.parametrize('band_type, step', [('octave', 1.0),
                                                 ('third_octave', 1.0 / 3.0)])
    def test_every_edge_is_exactly_the_half_step_from_its_centre(
            self, band_type, step):
        import math
        half = math.pow(2, step / 2)
        bands = self._bands(band_type)
        assert len(bands) > 3
        for low, centre, high in bands[:-1]:      # last high is clipped to fmax
            assert high == centre * half, (band_type, centre, high)
            assert low == centre / half, (band_type, centre, low)

    @pytest.mark.parametrize('band_type, step', [('octave', 1.0),
                                                 ('third_octave', 1.0 / 3.0)])
    def test_the_centres_advance_by_exactly_one_step(self, band_type, step):
        import math
        factor = math.pow(2, step)
        bands = self._bands(band_type)
        for lower, upper in zip(bands, bands[1:]):
            assert upper[1] == lower[1] * factor, (band_type, lower, upper)


class TestPPSDCarriesTheScalingItsLevelsAreStatedAgainst:
    """``scaling`` travels with the levels for the same reason ``ref`` does.

    ``ppsd`` takes ``scaling='density'|'spectrum'``, and the two mean different
    physical things: a density is per hertz, a spectrum is per band. A consumer
    that has to be told the scaling separately can be told the wrong one, which
    is exactly how ``plot_ppsd`` came to caption every histogram "/Hz".
    """

    @staticmethod
    def _x():
        rng = np.random.default_rng(0)
        return rng.standard_normal(48000), 48000.0

    def test_the_result_reports_the_scaling_it_was_computed_with(self):
        x, fs = self._x()
        for scaling in ('density', 'spectrum'):
            r = ppsd(x, fs, scaling=scaling, nperseg=1024)
            assert r.scaling == scaling, (scaling, r.scaling)

    def test_the_default_scaling_is_reported_not_left_blank(self):
        x, fs = self._x()
        assert ppsd(x, fs, nperseg=1024).scaling == 'density'

    def test_the_two_scalings_give_different_levels(self):
        # The control: if they were the same quantity there would be nothing
        # to carry. A density and a spectrum differ by the bin width.
        x, fs = self._x()
        d = ppsd(x, fs, scaling='density', nperseg=1024).mean_db
        sp = ppsd(x, fs, scaling='spectrum', nperseg=1024).mean_db
        assert not np.allclose(d, sp, atol=0.5), (d[:3], sp[:3])


class TestPPSDCarriesTheReferenceItsLevelsAreStatedAgainst:
    """``ppsd`` takes ``ref`` and reports every level as dB re ``ref**2``, but
    the result used not to carry that value, so every consumer had to guess it
    and the plotter hardcoded the package default.

    The gap is 120 dB wide: the same signal read against a Pa-based reference
    sits 120 dB below its µPa levels, and nothing in the returned tuple
    distinguished the two.
    """

    FS = 8000.0

    def _run(self, **kw):
        x = np.random.default_rng(0).standard_normal(int(4 * self.FS)) * 1e-3
        return ppsd(x, self.FS, seg_duration=1.0, nperseg=1024, **kw)

    def test_the_default_reference_is_reported(self):
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER
        assert self._run().ref == REFERENCE_PRESSURE_WATER

    @pytest.mark.parametrize('ref', [1.0, 1e-5, 20e-6])
    def test_a_non_default_reference_is_reported(self, ref):
        assert self._run(ref=ref).ref == ref

    def test_the_reference_tracks_a_real_120_db_move_in_the_levels(self):
        """Not a decorative field: the value it carries is what separates two
        results whose levels differ by 120 dB."""
        default = self._run()
        pascals = self._run(ref=1.0)
        shift = np.nanmean(pascals.mean_db - default.mean_db)
        assert shift == pytest.approx(-120.0, abs=1e-9)
        assert default.ref != pascals.ref

    def test_the_existing_fields_keep_their_positions(self):
        """``ref`` is appended, so anything indexing the tuple positionally —
        including result objects other suites build by hand — is unaffected."""
        r = self._run()
        assert r._fields[:7] == ('frequencies', 'level_edges', 'pdf',
                                 'mean_db', 'std_db', 'binwidth_db',
                                 'seg_duration')
        assert r._fields[7] == 'ref'
        from uacpy.acoustic_signal.analysis import PPSDResult
        built = PPSDResult(r.frequencies, r.level_edges, r.pdf, r.mean_db,
                           r.std_db, r.binwidth_db, r.seg_duration)
        assert built.ref == r.ref


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


def _boxcar_bin_energy(x, fs, nfft):
    """Per-bin exposure (Pa²·s) of the whole record, sel's own decomposition."""
    from scipy.signal import spectrogram, windows
    f, _t, Sxx = spectrogram(x, fs, window=windows.boxcar(nfft), noverlap=0,
                             nfft=nfft, detrend=False, scaling="density")
    return f, Sxx.sum(axis=1)


def test_sel_total_is_parseval_over_the_covered_band_only():
    """The band total accounts for every FFT bin inside
    ``[bands[0][0], bands[-1][2]]`` and nothing outside it.

    The docstring used to call the total Parseval-exact outright, which
    overstates it: a third-octave request snapped well below Nyquist keeps
    only the energy its bands span.
    """
    fs, nfft = 2000.0, 2000
    x = np.random.default_rng(0).standard_normal(4000)
    total = np.sum(x ** 2) / fs
    out = sel(x, fs, fmin=10, fmax=900, band_type="third_octave")
    lo, hi = out.bands[0][0], out.bands[-1][2]

    f, per_bin = _boxcar_bin_energy(x, fs, nfft)
    covered = per_bin[(f >= lo) & (f <= hi)].sum()
    assert out.sel_pa2s.sum() == pytest.approx(covered, rel=1e-9)
    # ... and that is a long way short of the whole record's exposure.
    assert out.sel_pa2s.sum() / total < 0.95


def test_sel_full_span_linear_drops_only_the_dc_bin():
    """A DC-to-Nyquist ``'linear'`` request keeps every bin but DC, whose
    band edges cannot reach (they must be > 0). The top band's edge is pinned
    to ``fmax`` so the Nyquist bin is not lost to float drift in the
    accumulated band width.
    """
    fs, nfft = 2000.0, 2000
    x = np.random.default_rng(0).standard_normal(4000)
    total = np.sum(x ** 2) / fs
    out = sel(x, fs, fmin=1e-12, fmax=fs / 2, band_type="linear", num_bands=50)
    assert out.bands[-1][2] == fs / 2

    f, per_bin = _boxcar_bin_energy(x, fs, nfft)
    assert total - out.sel_pa2s.sum() == pytest.approx(per_bin[f == 0.0].sum(),
                                                       rel=1e-9)
    # The top band is closed at fmax, so it carries the Nyquist bin too.
    top_lo = out.bands[-1][0]
    assert out.sel_pa2s[-1] == pytest.approx(
        per_bin[(f >= top_lo) & (f <= fs / 2)].sum(), rel=1e-9)


def test_ppsd_square_input_warns_and_takes_the_first_axis_as_time():
    """'the longer axis is time' cannot choose on a square input, so the
    first axis wins and the ambiguity is announced."""
    rng = np.random.default_rng(2)
    fs, n = 200.0, 64
    data = rng.standard_normal((n, n))
    with pytest.warns(UserWarning, match="square"):
        out = ppsd(data, fs, seg_duration=0.16, nperseg=16, lvlmin=-200,
                   lvlmax=200)
    columns = ppsd([data[:, i] for i in range(n)], fs, seg_duration=0.16,
                   nperseg=16, lvlmin=-200, lvlmax=200)
    assert np.allclose(out.mean_db, columns.mean_db)


def test_ppsd_segment_shorter_than_one_sample_is_diagnosed_as_such():
    """A ``seg_duration`` under one sample used to surface as an
    ``overlap_pct`` error, because the zero-length chunk made the step
    non-positive before anything checked the chunk itself."""
    from uacpy.core.exceptions import ConfigurationError
    x = np.random.default_rng(0).standard_normal(4000)
    with pytest.raises(ConfigurationError, match="seg_duration"):
        ppsd(x, 2000.0, seg_duration=1e-4)
    with pytest.raises(ConfigurationError, match="seg_duration"):
        ppsd(x, 2000.0, seg_duration=0.0)
    # The overlap check still owns its own case.
    with pytest.raises(ConfigurationError, match="overlap_pct"):
        ppsd(x, 2000.0, seg_duration=0.5, overlap_pct=100)


def test_ppsd_rejects_a_list_containing_a_non_1d_array():
    good = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(ConfigurationError, match='list element'):
        ppsd([good, np.ones((4, 4))], 1000.0, seg_duration=0.5)


class TestDecidecadeGuards:
    def _flat(self):
        f = np.linspace(50.0, 2000.0, 400)
        return np.ones_like(f), f

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_nonpositive_or_nonfinite_ref_raises(self, bad):
        p, f = self._flat()
        with pytest.raises(ConfigurationError,
                           match="ref must be > 0 Pa and finite"):
            decidecade_band_levels(p, f, ref=bad)

    def test_negative_psd_raises_typed(self):
        # A negative-power band failed the `power > 0` publication test and
        # came back as a silent NaN level.
        p, f = self._flat()
        p[10] = -1.0
        with pytest.raises(ConfigurationError, match="negative"):
            decidecade_band_levels(p, f)

    def test_flat_psd_gives_finite_levels_inside_support(self):
        p, f = self._flat()
        _, levels = decidecade_band_levels(p, f)
        assert np.isfinite(levels).any()


class TestSilentZerosAreAnnounced:
    def test_sel_warns_for_bands_holding_no_fft_bin(self):
        # 'linear' bands are used as given, so a fmax above Nyquist produces
        # whole bands that sum to exactly 0 Pa^2*s — indistinguishable from a
        # measured silence. The octave ladders clamp instead.
        rng = np.random.default_rng(0)
        x = rng.normal(size=4000)
        with pytest.warns(UserWarning, match="no FFT bin"):
            r = sel(x, 2000.0, band_type='linear', fmin=100.0, fmax=3000.0,
                    num_bands=6)
        assert np.all(r.sel_pa2s[2:] == 0.0)
        # Bands entirely below Nyquist stay silent about it.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sel(x, 2000.0, band_type='linear', fmin=100.0, fmax=900.0,
                num_bands=4)

    @pytest.mark.parametrize("estimator", ["psd", "ppsd", "spectrogram"])
    def test_complex_input_warns_about_the_two_sided_axis(self, estimator):
        # These three accept complex input and return an unsorted two-sided
        # axis, not the one-sided Pa^2/Hz their docstrings describe.
        rng = np.random.default_rng(0)
        z = rng.normal(size=2048) + 1j * rng.normal(size=2048)
        with pytest.warns(UserWarning, match="TWO-SIDED"):
            if estimator == "psd":
                f = psd(z, 1000.0, nperseg=128).frequencies
            elif estimator == "ppsd":
                f = ppsd(z, 1000.0, seg_duration=0.5, nperseg=128).frequencies
            else:
                f = spectrogram(z, 1000.0, nperseg=128).frequencies
        assert f.min() < 0.0                       # the two-sided axis
        assert not np.all(np.diff(f) > 0)          # and it is not sorted

    @pytest.mark.parametrize("estimator", [psd, spectrogram])
    def test_real_input_does_not_warn(self, estimator):
        x = np.random.default_rng(0).normal(size=2048)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimator(x, 1000.0, nperseg=128)


class TestPpsdOutOfWindowLevels:
    """When no PSD level lands inside ``[lvlmin, lvlmax]`` the pdf is
    all-NaN by construction; ``ppsd`` suppresses numpy's per-column 0/0
    RuntimeWarnings and says so once, naming the window and the measured
    level span."""

    def _signal(self):
        return np.sin(2 * np.pi * 100.0 * np.arange(4000) / 1000.0) * 1e9

    def test_out_of_window_levels_warn_once_and_raise_no_runtime(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ppsd(self._signal(), 1000.0, seg_duration=0.5,
                       nperseg=256)
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        typed = [str(w.message) for w in caught
                 if 'histogram window' in str(w.message)]
        assert len(typed) == 1
        assert 'lvlmin=0' in typed[0] and 'lvlmax=150' in typed[0]
        assert 'dB re ref²' in typed[0]
        assert np.all(np.isnan(res.pdf))
        assert np.all(np.isfinite(res.mean_db))

    def test_levels_inside_the_window_produce_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = ppsd(self._signal(), 1000.0, seg_duration=0.5,
                       nperseg=256, lvlmin=100, lvlmax=250)
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        assert not [w for w in caught
                    if 'histogram window' in str(w.message)]
        assert not np.all(np.isnan(res.pdf))


class TestDecidecadeSinglePositiveFrequencyGrid:
    """A grid whose only band support is one positive frequency (the
    two-sample rfftfreq grid) is reported in terms of the caller's
    ``frequencies`` argument, not the ``f_low``/``f_high`` arguments of the
    band helper it never called."""

    def test_error_names_the_callers_grid(self):
        with pytest.raises(ConfigurationError,
                           match='spans no decidecade band') as exc:
            decidecade_band_levels(np.ones(2), np.fft.rfftfreq(2))
        assert 'f_low' not in str(exc.value)

    def test_two_positive_frequencies_pass_the_guard(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            centers, levels = decidecade_band_levels(
                np.ones(3), np.fft.rfftfreq(4, d=1e-3))
        assert centers.size > 0
