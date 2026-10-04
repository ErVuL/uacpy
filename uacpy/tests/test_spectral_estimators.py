"""The six spectral estimators in ``uacpy.acoustic_signal.spectral``.

``welch`` and ``constant_q`` take ``scaling='density'`` or ``'spectrum'``,
``sound_exposure`` integrates band energy over the record, and each has a
``probabilistic_`` twin that returns the distribution of levels rather than
one level. What is pinned here is what a caller cannot see from a single
answer:

* **Every estimator says what it returned.** The result carries its scaling,
  its method and — for a banded estimate — its band type, so a figure or a
  downstream conversion reads the units off the answer instead of assuming
  them.
* **Each brings its own defaults.** The window, segment length and band ladder
  a statistic needs are its own, so no caller has to configure one estimator
  to get what another gives for free.
* **An exposure is the Welch bins summed.** ``sound_exposure`` is the Welch
  route integrated, checked bin for bin rather than by agreement to a
  tolerance, so the two can never drift into two different answers.
* **An energy is not a scaling of a bin estimator.** ``welch`` refuses
  ``scaling='exposure'`` by name rather than returning a number that is 0.9 to
  54 % wrong depending on the window.
* **Range and record are spelled the same way everywhere.** ``freq_min``/``freq_max``
  and ``integration_time`` mean one thing across all six doors.

The plotters that draw these estimates are exercised in
``test_visualization.py``; the constant-Q transform's own theory is in
``test_constant_q_transform.py``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch as _scipy_welch
from uacpy.acoustic_signal.spectral import (
    ProbabilisticSpectralEstimate, constant_q, probabilistic_constant_q,
    probabilistic_sound_exposure, probabilistic_welch, sound_exposure, welch,
)
import warnings

from uacpy.acoustic_signal.bands import decidecade_band_levels
from uacpy.acoustic_signal.timefreq import spectrogram
from uacpy.core.acoustics import power_to_dB
from uacpy.core.exceptions import ConfigurationError

#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]
from uacpy.visualization.plots.signal import (
    plot_psd, plot_ppsd, plot_sel)
from uacpy.tests.conftest import recorded_warnings


def _band_exposure(data, sample_rate, **options):
    """The ISO-band sound exposure: the estimator's ``'exposure'`` scaling
    reported on a band ladder. Returns a ``SpectralEstimate`` whose ``power``
    is Pa²·s per band and whose ``bands`` are the ``(low, centre, high)``
    edges those values sit on.
    """
    return sound_exposure(data, sample_rate, **options)




def test_psd_matches_welch():
    # These welch arguments are welch()'s effective defaults — window="hann",
    # nperseg=8192, and the noverlap=None that scipy resolves to nperseg//2.
    # Spelling them out makes the default segmentation part of the contract:
    # changing any of them silently changes every caller's spectrum.
    x = np.random.default_rng(0).standard_normal(48000) * 3.0
    f0, p0 = _scipy_welch(x, 48000.0, window="hann", nperseg=8192, noverlap=4096,
                   scaling="density")
    f, p = welch(x, 48000.0)
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
            f, pd_ = welch(x, fs, window=win, nperseg=nper,
                                       )
            _, ps_ = welch(x, fs, scaling='spectrum', window=win, nperseg=nper,
                                       )
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
    # Both windows named: the ratio IS the window's bandwidth, and the two
    # scalings no longer default to the same window (a spectrum defaults to
    # flattop, whose 3.77-bin bandwidth would be the answer instead).
    f, pd_ = welch(x, fs, nperseg=nper, window="hann",
                               )
    _, ps_ = welch(x, fs, scaling='spectrum', nperseg=nper, window="hann",
                               )
    np.testing.assert_allclose(ps_[1:-1] / pd_[1:-1], 1.5 * fs / nper,
                               rtol=1e-12)


def test_psd_hann_worst_case_scalloping_is_1_42_dB():
    """A tone half a bin off centre reads ``8/(3*pi)`` of the on-bin amplitude
    through a hann window: ``20*log10(8/(3*pi)) = -1.4236`` dB, the worst-case
    1.42 dB scalloping loss. On a bin centre ``scaling='spectrum'`` reads the
    full ``A**2/2``.

    This loss is why :func:`welch` defaults to ``flattop`` instead,
    where the same half-bin tone reads about 0.01 dB low;
    :func:`test_a_flattop_spectrum_reads_a_tone_between_bins` measures that.
    Hann is named here because it is this test's subject."""
    fs, nper = 1000.0, 256
    t = np.arange(4 * nper) / fs
    f_on = 32 * fs / nper
    f_off = 32.5 * fs / nper
    _, p_on = welch(np.cos(2 * np.pi * f_on * t), fs, scaling='spectrum',
                             nperseg=nper, window="hann")
    _, p_off = welch(np.cos(2 * np.pi * f_off * t), fs, scaling='spectrum',
                              nperseg=nper, window="hann")
    assert p_on.max() == pytest.approx(0.5, rel=1e-6)
    loss_dB = 10 * np.log10(p_off.max() / p_on.max())
    assert loss_dB == pytest.approx(20 * np.log10(8 / (3 * np.pi)), abs=1e-3)
    assert loss_dB == pytest.approx(-1.42, abs=0.01)


def test_ppsd_function():
    x = np.random.default_rng(0).standard_normal(48000 * 4)
    r = probabilistic_welch(x, 48000.0, segment_duration=1.0)
    assert isinstance(r, ProbabilisticSpectralEstimate)
    assert r.pdf.shape[1] == r.frequencies.size
    fig, ax = plot_ppsd(r)
    assert ax.collections
    plt.close(fig)


def test_sel_parseval():
    fs, T = 48000.0, 5.0
    t = np.arange(int(T * fs)) / fs
    x = 2.5 * np.sin(2 * np.pi * 1000.0 * t)
    est = _band_exposure(x, fs)
    s, bands = est.power, est.bands
    assert abs(np.sum(s) / (np.sum(x ** 2) / fs) - 1.0) < 0.01
    fig, ax = plot_sel(s, bands, duration=T)
    assert ax.patches
    plt.close(fig)


def test_sel_conserves_energy_for_any_nfft():
    """``sound_exposure``'s contract is Parseval: the summed band exposure
    equals
    ``sum(p**2)/fs`` in Pa²·s. Bin width and segment duration cancel
    (``(fs/nfft) * (nfft/fs) == 1``), so it must hold for any ``nfft``."""
    from scipy.signal import butter, sosfiltfilt
    fs = 48000.0
    rng = np.random.default_rng(0)
    sos = butter(6, [50 / (fs / 2), 15000 / (fs / 2)], btype='band', output='sos')
    x = sosfiltfilt(sos, rng.standard_normal(int(4 * fs)))
    truth = float(np.sum(x ** 2) / fs)
    for nfft in (None, 24000, 96000):
        assert _band_exposure(x, fs, nperseg=nfft).power.sum() == pytest.approx(truth,
                                                                     rel=1e-4)


def test_the_default_ladder_is_the_base_10_decidecade_one():
    """``band_type='decidecade'`` is the default because it is the ladder the
    standards are written on: IEC 61260-1 / ISO 18405 centres are
    ``1000*10**(n/10)`` with edges ``10**(+/-1/20)`` either side."""
    fs = 48000.0
    est = _band_exposure(np.zeros(int(fs)), fs)
    assert est.band_type == 'decidecade'
    centres = est.frequencies
    assert np.allclose(centres[1:] / centres[:-1], 10 ** 0.1)
    # The nominal reporting range, 10 Hz to 20 kHz, read off the ladder.
    assert centres[0] == pytest.approx(10.0)
    assert centres[-1] == pytest.approx(10 ** 4.3)
    for low, centre, high in est.bands:
        assert high / centre == pytest.approx(10 ** 0.05)
        assert centre / low == pytest.approx(10 ** 0.05)


def test_the_octave_ladder_is_the_base_10_one():
    """IEC 61260-1:2014 octaves are base 10: centres ``1000*10**(3k/10)``,
    so the band four octaves above 1 kHz is 15.85 kHz, not 16 kHz."""
    fs = 48000.0
    bands = np.asarray(_band_exposure(np.zeros(int(fs)), fs,
                                      band_type='octave').bands)
    np.testing.assert_allclose(bands[1:, 1] / bands[:-1, 1], 10 ** 0.3,
                               rtol=1e-12)
    np.testing.assert_allclose(bands[:, 2] / bands[:, 1], 10 ** 0.15,
                               rtol=1e-12)
    assert bands[-1, 1] == pytest.approx(10 ** 4.2, rel=1e-12)


def test_there_is_one_name_for_the_base_10_third_octave():
    """The base-10 third-octave is the decidecade, so ``'third_octave'`` is
    not a ladder; the refusal lists the ones there are."""
    with pytest.raises(ConfigurationError,
                       match="unknown band_type 'third_octave'"):
        _band_exposure(np.zeros(8000), 8000.0, band_type='third_octave')
    import uacpy.acoustic_signal as sig
    assert not hasattr(sig, 'third_octave_bands')


class TestEveryLadderComesFromOneBuilder:
    """The three public ladders, :func:`standard_bands` and
    :func:`sound_exposure` build their bands with one fractional-octave
    ladder anchored at 1 kHz, base 10 (IEC 61260-1): ``'decidecade'`` and
    ``'octave'``, with exact base-2 octaves on request. Edges are compared
    with ``==`` here: an approx comparison would not notice a rewrite that
    moved them by an ULP."""

    FS = 48000.0

    @pytest.mark.parametrize('freq_min, freq_max', [
        (8.9125, 22387.0), (10.0, 25000.0), (100.0, 1000.0), (0.5, 3.0),
        (47.0, 48000.0), (1e-3, 1e6)])
    def test_decidecade_bands_are_the_base_10_formula_to_the_last_bit(
            self, freq_min, freq_max):
        """Centres ``1000 * 10**(n/10)`` and edges ``centre * 10**(±1/20)``,
        every band overlapping the range. The builder writes them as the
        IEC 61260-1 ``G = 10**(3/10)`` ladder at three bands per octave; its
        exponents ``3n/30`` and ``3/60`` are the correctly rounded ``n/10``
        and ``1/20``, so the doubles are these."""
        from uacpy.acoustic_signal import decidecade_bands
        n = np.arange(int(np.floor(10.0 * np.log10(freq_min / 1000.0))),
                      int(np.ceil(10.0 * np.log10(freq_max / 1000.0))) + 1)
        centres = 1000.0 * 10.0 ** (n / 10.0)
        lower = centres * 10.0 ** (-1.0 / 20.0)
        upper = centres * 10.0 ** (1.0 / 20.0)
        keep = (upper >= freq_min) & (lower <= freq_max)
        for got, want in zip(decidecade_bands(freq_min, freq_max),
                             (lower[keep], centres[keep], upper[keep])):
            np.testing.assert_array_equal(got, want)

    def test_the_base_2_octaves_are_exact_octaves(self):
        """``octave_bands(base=2)``: centres ``1000 * 2**k``, edges
        ``centre * 2**(±1/2)`` (Pierce, *Acoustics*, §2.2), consecutive,
        1 kHz among them, and every band overlapping the range."""
        from uacpy.acoustic_signal import octave_bands
        lower, centres, upper = octave_bands(20.0, 20000.0, base=2)
        k = np.round(np.log2(centres / 1000.0))
        np.testing.assert_array_equal(np.diff(k), 1.0)
        np.testing.assert_array_equal(centres, 1000.0 * 2.0 ** k)
        np.testing.assert_array_equal(lower, centres * 2.0 ** -0.5)
        np.testing.assert_array_equal(upper, centres * 2.0 ** 0.5)
        assert 1000.0 in centres
        assert lower[0] < 20.0 <= upper[0]
        assert lower[-1] <= 20000.0 < upper[-1]

    def test_an_octave_is_three_decidecades(self):
        """The base-10 octave ``k`` is centred on decidecade ``3k`` and
        spans decidecades ``3k-1`` to ``3k+1``, edge to edge."""
        from uacpy.acoustic_signal import decidecade_bands, octave_bands
        lower, centres, upper = octave_bands(20.0, 20000.0)
        d_lower, d_centres, d_upper = decidecade_bands(lower[0], upper[-1])
        for lo, c, hi in zip(lower, centres, upper):
            i = int(np.argmin(np.abs(d_centres - c)))
            assert c == pytest.approx(d_centres[i], rel=1e-14)
            assert lo == pytest.approx(d_lower[i - 1], rel=1e-14)
            assert hi == pytest.approx(d_upper[i + 1], rel=1e-14)
        assert 1000.0 in centres

    def test_the_base_10_and_base_2_octaves_part_by_a_quarter_percent_a_step(
            self):
        """Four octaves above 1 kHz: 15.85 kHz on the IEC ladder, 16 kHz on
        the exact one."""
        from uacpy.acoustic_signal import octave_bands
        assert octave_bands(15000.0, 15500.0)[1][-1] == pytest.approx(
            10.0 ** 4.2, rel=1e-15)
        assert octave_bands(15000.0, 17000.0, base=2)[1][-1] == 16000.0

    @pytest.mark.parametrize('base', [3, 2.5, '10', None])
    def test_octave_bands_build_base_10_or_base_2_only(self, base):
        from uacpy.acoustic_signal import octave_bands
        with pytest.raises(ConfigurationError, match='base must be 10'):
            octave_bands(20.0, 200.0, base=base)

    @pytest.mark.parametrize('band_type', ['decidecade', 'octave', 'linear'])
    def test_standard_bands_are_the_bands_sound_exposure_reports_on(
            self, band_type):
        from uacpy.acoustic_signal import standard_bands
        est = _band_exposure(np.zeros(int(self.FS)), self.FS,
                             band_type=band_type)
        want = np.column_stack(standard_bands(
            8.9125, 22387.0, band_type=band_type, sample_rate=self.FS,
            n_bands=30))
        np.testing.assert_array_equal(np.asarray(est.bands), want)
        np.testing.assert_array_equal(est.frequencies, want[:, 1])

    def test_standard_bands_select_decidecades_on_their_centre(self):
        """``freq_max=22387`` keeps the 20 kHz band although its exact upper
        edge is 22387.21 Hz; a sample rate whose Nyquist cuts that band
        drops it, and without one it stays."""
        from uacpy.acoustic_signal import standard_bands
        _, centres, upper = standard_bands(8.9125, 22387.0)
        assert centres[0] == pytest.approx(10.0)
        assert centres[-1] == pytest.approx(10.0 ** 4.3)
        assert upper[-1] > 22387.0
        _, cut, _ = standard_bands(8.9125, 22387.0, sample_rate=44100.0)
        np.testing.assert_array_equal(cut, centres[:-1])

    def test_standard_bands_run_from_the_octave_holding_each_end(self):
        from uacpy.acoustic_signal import standard_bands
        lower, _, upper = standard_bands(10.0, 3000.0, band_type='octave')
        assert lower[0] <= 10.0 < upper[0]
        assert lower[-1] <= 3000.0 < upper[-1]
        lower, _, upper = standard_bands(10.0, 3000.0, band_type='octave',
                                         sample_rate=4000.0)
        # The highest band wholly under the 2 kHz Nyquist, and no higher.
        assert upper[-1] <= 2000.0
        assert upper[-1] * (upper[-1] / lower[-1]) > 2000.0

    def test_standard_bands_refuse_what_they_cannot_build(self):
        from uacpy.acoustic_signal import standard_bands
        with pytest.raises(ConfigurationError, match='unknown band_type'):
            standard_bands(10.0, 100.0, band_type='fifth_octave')
        with pytest.raises(ConfigurationError,
                           match='n_bands must be positive'):
            standard_bands(10.0, 100.0, band_type='linear')
        with pytest.raises(ConfigurationError,
                           match='require freq_min > 0 and freq_max > freq_min'):
            standard_bands(100.0, 10.0)

    def test_band_levels_on_the_decidecade_ladder_are_decidecade_band_levels(
            self):
        from uacpy.acoustic_signal import band_levels, decidecade_band_levels
        f = np.linspace(0.0, 4000.0, 4097)
        psd = np.random.default_rng(7).exponential(1e-10, f.size)
        for got, want in zip(band_levels(psd, f),
                             decidecade_band_levels(psd, frequencies=f)):
            np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize('band_type, ladder', [
        ('decidecade', 'decidecade_bands'), ('octave', 'octave_bands')])
    def test_band_levels_integrate_a_flat_psd_over_each_band_width(
            self, band_type, ladder):
        """A flat PSD gives ``10*log10(P * width / ref**2)`` in every band
        the grid covers; the two bands overhanging its ends are ``nan``."""
        import uacpy.acoustic_signal as sig
        f = np.linspace(0.0, 8000.0, 16001)
        centres, levels = sig.band_levels(np.full(f.size, 1e-10), f,
                                          band_type=band_type)
        lower, want_centres, upper = getattr(sig, ladder)(0.5, 8000.0)
        np.testing.assert_array_equal(centres, want_centres)
        assert np.isnan(levels[0]) and np.isnan(levels[-1])
        np.testing.assert_allclose(
            levels[1:-1],
            10.0 * np.log10(1e-10 * (upper - lower)[1:-1] / 1e-12),
            atol=1e-9, rtol=0)

    def test_band_levels_carry_the_ladder_they_are_stated_on(self):
        """The levels come back as a ``BandLevels`` that still unpacks to
        ``(centres, levels)`` and carries the edges, the ladder and the
        reference, through a copy and a pickle; its unit names the ref."""
        import copy
        import pickle
        import uacpy.acoustic_signal as sig
        f = np.linspace(0.0, 8000.0, 16001)
        result = sig.band_levels(np.full(f.size, 1e-10), f,
                                 band_type='octave', ref=1.0)
        assert isinstance(result, sig.BandLevels)
        centres, levels = result
        lower, want, upper = sig.octave_bands(0.5, 8000.0)
        np.testing.assert_array_equal(centres, want)
        np.testing.assert_array_equal(result.lower, lower)
        np.testing.assert_array_equal(result.upper, upper)
        for twin in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
            assert (twin.band_type, twin.ref) == ('octave', 1.0)
            np.testing.assert_array_equal(twin.upper, upper)
        assert result.units == {'centres': 'Hz', 'levels': 'dB re 1 Pa²'}
        assert sig.decidecade_band_levels(np.full(f.size, 1e-10),
                                          frequencies=f).units['levels'] == 'dB re 1 µPa²'

    @pytest.mark.parametrize('ref', [1e-6, 1.0])
    def test_band_levels_draw_against_their_own_reference(self, ref):
        """``.plot()`` hands the levels to ``plot_band_levels`` with the
        ``ref`` they are stated re, so the axis names that reference."""
        from uacpy.acoustic_signal import band_levels
        f = np.linspace(0.0, 8000.0, 16001)
        result = band_levels(np.full(f.size, 1e-10), f, ref=ref)
        fig, ax = result.plot()
        try:
            # The micro prefix appears exactly when the levels are re 1 µPa.
            assert ('µPa' in ax.get_ylabel()) == (ref == 1e-6)
            assert len(ax.patches) == result.centres.size
        finally:
            plt.close(fig)

    def test_band_levels_refuse_a_ladder_without_standard_edges(self):
        from uacpy.acoustic_signal import band_levels
        f = np.linspace(0.0, 100.0, 101)
        with pytest.raises(ConfigurationError,
                           match='has no standard band edges'):
            band_levels(np.ones(f.size), f, band_type='linear')


class TestPPSDCarriesTheScalingItsLevelsAreStatedAgainst:
    """``scaling`` travels with the levels for the same reason ``ref`` does.

    ``probabilistic_welch`` and its spectrum twin mean
    different
    physical things: a density is per hertz, a spectrum is per band. A consumer
    that has to be told the scaling separately can be told the wrong one, which
    is exactly how ``plot_ppsd`` came to caption every histogram "/Hz".
    """

    @staticmethod
    def _x():
        rng = np.random.default_rng(0)
        return rng.standard_normal(48000), 48000.0

    def test_each_door_reports_the_scaling_its_name_promises(self):
        x, fs = self._x()
        for scaling in ('density', 'spectrum'):
            r = probabilistic_welch(x, fs, nperseg=1024, scaling=scaling)
            assert r.scaling == scaling, (scaling, r.scaling)

    def test_the_default_scaling_is_reported_not_left_blank(self):
        x, fs = self._x()
        assert probabilistic_welch(x, fs, nperseg=1024).scaling == 'density'

    def test_the_two_scalings_give_different_levels(self):
        # The control: if they were the same quantity there would be nothing
        # to carry. A density and a spectrum differ by the bin width.
        x, fs = self._x()
        d = probabilistic_welch(x, fs, nperseg=1024).mean_dB
        sp = probabilistic_welch(x, fs, scaling='spectrum', nperseg=1024,
                                          window='hann').mean_dB
        assert not np.allclose(d, sp, atol=0.5), (d[:3], sp[:3])


class TestPPSDCarriesTheReferenceItsLevelsAreStatedAgainst:
    """The histogram estimators take ``ref`` and report every level as dB re
    ``ref**2``, but
    the result used not to carry that value, so every consumer had to guess it
    and the plotter hardcoded the package default.

    The gap is 120 dB wide: the same signal read against a Pa-based reference
    sits 120 dB below its µPa levels, and nothing in the returned tuple
    distinguished the two.
    """

    FS = 8000.0

    def _run(self, **kw):
        x = np.random.default_rng(0).standard_normal(int(4 * self.FS)) * 1e-3
        return probabilistic_welch(x, self.FS, segment_duration=1.0, nperseg=1024, **kw)

    def test_the_default_reference_is_reported(self):
        from uacpy.core.constants import REFERENCE_PRESSURE_WATER
        assert self._run().ref == REFERENCE_PRESSURE_WATER

    @pytest.mark.parametrize('ref', [1.0, 1e-5, 20e-6])
    def test_a_non_default_reference_is_reported(self, ref):
        assert self._run(ref=ref).ref == ref

    def test_the_reference_tracks_a_real_120_dB_move_in_the_levels(self):
        """Not a decorative field: the value it carries is what separates two
        results whose levels differ by 120 dB."""
        default = self._run()
        pascals = self._run(ref=1.0)
        shift = np.nanmean(pascals.mean_dB - default.mean_dB)
        assert shift == pytest.approx(-120.0, abs=1e-9)
        assert default.ref != pascals.ref

    def test_the_existing_fields_keep_their_positions(self):
        """``ref`` is appended, so anything indexing the tuple positionally —
        including result objects other suites build by hand — is unaffected."""
        r = self._run()
        # The tuple is the measurement; everything that says what it MEANS is
        # an attribute, and a result built without them takes the defaults.
        assert r._fields == ('frequencies', 'level_edges', 'pdf')
        from uacpy.acoustic_signal.spectral import (
            ProbabilisticSpectralEstimate,
        )
        built = ProbabilisticSpectralEstimate(r.frequencies, r.level_edges,
                                              r.pdf)
        assert built.ref == r.ref
        assert (built.scaling, built.method) == ('density', 'welch')
        assert (built.segment_duration, built.bands) == (None, None)


def test_ppsd_columns_are_densities_with_empty_bins_at_zero():
    """Each frequency column integrates to 1 over the level axis with a plain
    sum: a level never observed is a density of 0 (it was NaN, 95 % of a
    white-noise histogram, and pdf.sum() was NaN). The plotter blanks it."""
    fs = 8000.0
    rng = np.random.default_rng(1)
    r = probabilistic_welch(rng.standard_normal(int(30 * fs)) * 1e-3, fs, segment_duration=1.0,
             nperseg=1024, noverlap=512)
    integral = r.pdf.sum(axis=0) * r.level_step_dB
    assert np.allclose(integral, 1.0)
    assert (r.pdf == 0).any() and not np.isnan(r.pdf).any()

    centres = (r.level_edges[:-1] + r.level_edges[1:]) / 2
    first_moment = (r.pdf * centres[:, None]).sum(axis=0) * r.level_step_dB
    band = (r.frequencies > 200) & (r.frequencies < 3500)
    assert np.abs(r.mean_dB[band] - first_moment[band]).max() < r.level_step_dB


class TestSELBandGridIsAnchoredAt1kHz:
    """IEC 61260-1 anchors both band systems at 1 kHz — Pierce: "1, 10, 100,
    1000, 10,000 Hz … are also standard 1/3-octave-band f_o's". Snapping the
    ladder to the caller's ``freq_min`` instead made the grid move with the
    request: ``freq_min=8.9125`` and ``freq_min=10.0`` produced disjoint, interleaved
    centres, and the nearest centre to 1 kHz was 1024 Hz (+2.4 %) or 912.3 Hz
    (-8.8 %) depending on it. ``decidecade_bands`` in the same package already
    anchors correctly; the band estimator was the outlier."""

    FS = 48000

    def _centres(self, **kw):
        y = np.zeros(self.FS)
        return np.array([b[1] for b in _band_exposure(y, self.FS, **kw).bands])

    @pytest.mark.parametrize('band_type', ['decidecade', 'octave'])
    def test_one_kilohertz_is_a_band_centre(self, band_type):
        assert np.isclose(self._centres(band_type=band_type), 1000.0).any()

    def test_octave_ladder_is_not_powers_of_two(self):
        # 1024 Hz was reported where a soundscape table expects 1000.
        oc = self._centres(band_type='octave')
        assert not np.isclose(oc, 1024.0).any()

    @pytest.mark.parametrize('freq_min', [10.0, 12.0, 20.0, 25.0])
    def test_grids_nest_instead_of_interleaving(self, freq_min):
        # The discriminating property: changing freq_min may drop bands off the
        # bottom but must never shift the ladder. Before the fix these sets
        # were disjoint from the default one.
        base = self._centres()
        got = self._centres(freq_min=freq_min)
        assert np.isclose(got[:, None], base[None, :], rtol=1e-9).any(axis=1).all()

    @pytest.mark.parametrize('fs', [2000, 8000, 48000])
    def test_highest_band_stays_below_nyquist(self, fs):
        y = np.zeros(fs)
        highs = np.array([b[2] for b in _band_exposure(y, fs).bands])
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
        a = probabilistic_welch(x, self.FS, segment_duration=1.0, nperseg=2048, noverlap=0)
        b = probabilistic_welch(x, self.FS, segment_duration=1.0, nperseg=2048, noverlap=1536)
        assert not np.array_equal(np.nan_to_num(a.pdf), np.nan_to_num(b.pdf))

    def test_unfittable_noverlap_warns_and_falls_back(self):
        x = self._sig()
        with pytest.warns(UserWarning, match="noverlap"):
            r = probabilistic_welch(x, self.FS, segment_duration=0.1, noverlap=4096)
        assert np.isfinite(np.nansum(r.pdf))

    def test_default_matches_half_nperseg(self):
        x = self._sig()
        d = probabilistic_welch(x, self.FS, segment_duration=1.0, nperseg=2048)
        e = probabilistic_welch(x, self.FS, segment_duration=1.0, nperseg=2048, noverlap=1024)
        np.testing.assert_array_equal(np.nan_to_num(d.pdf), np.nan_to_num(e.pdf))


def test_a_batch_size_that_does_not_hold_whole_segments_warns():
    fs = 8000
    x = np.sin(2 * np.pi * 1000.5 * np.arange(fs * 4) / fs)
    with pytest.warns(UserWarning, match="multiple of nperseg"):
        _band_exposure(x, fs, batch_size=fs + fs // 3)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        _band_exposure(x, fs, batch_size=2 * fs)          # aligned: silent


def test_sel_rejects_complex_input():
    from uacpy.core.exceptions import ConfigurationError
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    with pytest.raises(ConfigurationError, match="complex"):
        _band_exposure(np.exp(2j * np.pi * 1000.0 * t), fs)


def test_sel_splits_a_bin_on_a_band_edge_by_its_overlap():
    """A bin is an interval ``f ± df/2``; one centred on a band edge lies half
    in each band. A tone on the top edge freq_max therefore puts half its bin in
    the last band and half above the ladder; a tone on an interior edge puts
    half in each neighbour, and the two halves sum to the tone."""
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    tone = lambda f: np.sqrt(2.0) * np.cos(2 * np.pi * f * t)   # 1 Pa² · s
    out = _band_exposure(tone(3000.0), fs, band_type="linear", freq_min=100,
                         freq_max=3000, n_bands=8)
    assert out.power[-1] == pytest.approx(0.5, rel=1e-9)
    assert out.power.sum() == pytest.approx(0.5, rel=1e-9)
    edge = out.bands[3][2]                     # 1550 Hz, on a bin centre
    assert edge == 1550.0
    inner = _band_exposure(tone(edge), fs, band_type="linear", freq_min=100,
                           freq_max=3000, n_bands=8)
    assert inner.power[3] == pytest.approx(0.5, rel=1e-9)
    assert inner.power[4] == pytest.approx(0.5, rel=1e-9)


def test_sel_rejects_nonpositive_sample_rate():
    from uacpy.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="sample_rate"):
        _band_exposure(np.ones(100), 0.0)


def _overlap_weights(f, df, fs, low, high):
    """Fraction of each rfft bin's interval inside ``[low, high]``: bin k>0
    covers ``[f-df/2, f+df/2]`` (Nyquist its lower half), DC none."""
    lo = np.maximum(f - df / 2, 0.0)
    hi = np.minimum(f + df / 2, fs / 2)
    w = np.clip(np.minimum(hi, high) - np.maximum(lo, low), 0.0, None) / (hi - lo)
    w[f == 0.0] = 0.0
    return w


def _boxcar_bin_energy(x, fs, nfft):
    """Per-bin exposure (Pa²·s) of the whole record, sel's own decomposition."""
    from scipy.signal import spectrogram, windows
    f, _t, Sxx = spectrogram(x, fs, window=windows.boxcar(nfft), noverlap=0,
                             nfft=nfft, detrend=False, scaling="density")
    return f, Sxx.sum(axis=1)


def test_sel_total_is_parseval_over_the_covered_band_only():
    """The band total accounts for exactly the part of the FFT bins inside
    ``[bands[0][0], bands[-1][2]]`` and nothing outside it: an octave
    request clamped well below Nyquist keeps only the energy its bands
    span."""
    fs, nfft = 2000.0, 2000
    x = np.random.default_rng(0).standard_normal(4000)
    total = np.sum(x ** 2) / fs
    out = _band_exposure(x, fs, freq_min=10, freq_max=900, band_type="octave")
    lo, hi = out.bands[0][0], out.bands[-1][2]

    f, per_bin = _boxcar_bin_energy(x, fs, nfft)
    covered = per_bin @ _overlap_weights(f, fs / nfft, fs, lo, hi)
    assert out.power.sum() == pytest.approx(covered, rel=1e-9)
    # ... and that is a long way short of the whole record's exposure.
    assert out.power.sum() / total < 0.95


def test_sel_full_span_linear_drops_only_the_dc_bin():
    """A DC-to-Nyquist ``'linear'`` request keeps every bin but DC, whose
    band edges cannot reach (they must be > 0). The top band's edge is pinned
    to ``freq_max`` so the Nyquist bin is not lost to float drift in the
    accumulated band width.
    """
    fs, nfft = 2000.0, 2000
    x = np.random.default_rng(0).standard_normal(4000)
    total = np.sum(x ** 2) / fs
    out = _band_exposure(x, fs, freq_min=1e-12, freq_max=fs / 2, band_type="linear", n_bands=50)
    assert out.bands[-1][2] == fs / 2

    f, per_bin = _boxcar_bin_energy(x, fs, nfft)
    assert total - out.power.sum() == pytest.approx(per_bin[f == 0.0].sum(),
                                                       rel=1e-9)
    # The top band is closed at freq_max, so it carries the Nyquist bin too.
    top_lo = out.bands[-1][0]
    assert out.power[-1] == pytest.approx(
        per_bin @ _overlap_weights(f, fs / nfft, fs, top_lo, fs / 2),
        rel=1e-9)


def test_ppsd_takes_the_time_axis_by_name_and_never_guesses():
    """``axis`` names the time axis, under welch's rule: a read_wav-shaped
    ``(n_samples, n_channels)`` block is refused at the default and pooled
    channel by channel at ``axis=0``; channel-first data is read along its
    last axis."""
    from uacpy.core.exceptions import ConfigurationError
    rng = np.random.default_rng(2)
    fs = 200.0
    data = rng.standard_normal((800, 3))
    kw = dict(segment_duration=0.16, nperseg=16, level_min_dB=-200, level_max_dB=200)
    with pytest.raises(ConfigurationError, match="axis=0"):
        probabilistic_welch(data, fs, **kw)
    by_axis = probabilistic_welch(data, fs, axis=0, **kw)
    columns = probabilistic_welch([data[:, i] for i in range(3)], fs, **kw)
    assert np.array_equal(by_axis.mean_dB, columns.mean_dB)
    channel_first = probabilistic_welch(data.T, fs, **kw)
    assert np.array_equal(channel_first.mean_dB, columns.mean_dB)


def test_ppsd_integration_time_trims_the_time_axis_of_a_block():
    """The record is trimmed along the axis ``axis`` names, not along the
    channels: 2 s of a 4 s ``(n_samples, n_channels)`` block is the first
    2 s of every channel."""
    rng = np.random.default_rng(3)
    fs = 200.0
    data = rng.standard_normal((800, 2))
    kw = dict(segment_duration=0.16, nperseg=16, level_min_dB=-200, level_max_dB=200)
    trimmed = probabilistic_welch(data, fs, axis=0, integration_time=2.0,
                                  **kw)
    first = probabilistic_welch(data[:400], fs, axis=0, **kw)
    assert np.array_equal(trimmed.mean_dB, first.mean_dB)


@pytest.mark.parametrize('estimator', [probabilistic_welch,
                                       probabilistic_constant_q])
def test_ppsd_integration_time_refuses_records_of_different_lengths(
        estimator):
    """Records of different lengths share no time axis to trim; the
    refusal is typed and names the knob. Equal lengths still trim as the
    block they stack into."""
    rng = np.random.default_rng(4)
    fs = 2000.0
    kw = dict(level_min_dB=-200, level_max_dB=200)
    if estimator is probabilistic_welch:
        kw.update(segment_duration=0.16, nperseg=64)
    ragged = [rng.standard_normal(8000), rng.standard_normal(6000)]
    with pytest.raises(ConfigurationError,
                       match=r"integration_time=.*different shapes"):
        estimator(ragged, fs, integration_time=1.0, **kw)
    equal = [rng.standard_normal(8000), rng.standard_normal(8000)]
    trimmed = estimator(equal, fs, integration_time=2.0, **kw)
    first = estimator([s[:4000] for s in equal], fs, **kw)
    assert np.array_equal(trimmed.mean_dB, first.mean_dB)


def test_ppsd_segment_shorter_than_one_sample_is_diagnosed_as_such():
    """A ``segment_duration`` under one sample used to surface as an
    ``segment_overlap_percent`` error, because the zero-length chunk made the step
    non-positive before anything checked the chunk itself."""
    from uacpy.core.exceptions import ConfigurationError
    x = np.random.default_rng(0).standard_normal(4000)
    with pytest.raises(ConfigurationError, match="segment_duration"):
        probabilistic_welch(x, 2000.0, segment_duration=1e-4)
    with pytest.raises(ConfigurationError, match="segment_duration"):
        probabilistic_welch(x, 2000.0, segment_duration=0.0)
    # The overlap check still owns its own case.
    with pytest.raises(ConfigurationError, match="segment_overlap_percent"):
        probabilistic_welch(x, 2000.0, segment_duration=0.5, segment_overlap_percent=100)


def test_ppsd_rejects_a_list_containing_a_non_1d_array():
    good = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(ConfigurationError, match='list element'):
        probabilistic_welch([good, np.ones((4, 4))], 1000.0, segment_duration=0.5)


class TestHistogramWindowLossIsNamed:
    """Each ``pdf`` column is normalised over the levels inside
    ``[level_min_dB, level_max_dB]``, so a window clipping more than 5 % of a column's
    levels says so rather than drawing the survivors as the whole."""

    def test_the_threshold_is_more_than_five_percent(self):
        from uacpy.acoustic_signal.spectral import _warn_levels_outside_window
        edges = np.arange(0.0, 11.0)
        inside = np.full(20, 5.0)
        one_out, two_out = inside.copy(), inside.copy()
        one_out[0] = 20.0
        two_out[:2] = 20.0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _warn_levels_outside_window('t', [one_out], np.array([100.0]),
                                        edges)
        with pytest.warns(UserWarning,
                          match=r"1 of 1 frequency column.*10 % at 100 Hz"):
            _warn_levels_outside_window('t', [two_out], np.array([100.0]),
                                        edges)

    def _noise(self):
        return np.random.default_rng(0).standard_normal(16000)

    def test_welch_histogram_names_a_clipping_window(self):
        x = self._noise()
        full = probabilistic_welch(x, 2000.0, segment_duration=0.5, nperseg=256)
        middle = float(np.nanmedian(full.mean_dB))
        with pytest.warns(UserWarning, match='outside the histogram window'):
            probabilistic_welch(x, 2000.0, segment_duration=0.5, nperseg=256,
                                level_max_dB=middle)

    def test_constant_q_histogram_names_a_clipping_window(self):
        from uacpy.acoustic_signal import probabilistic_constant_q
        x = self._noise()
        kw = dict(freq_min=100.0, freq_max=800.0, bins_per_octave=3)
        full = probabilistic_constant_q(x, 2000.0, **kw)
        middle = float(np.nanmedian(full.mean_dB))
        with pytest.warns(UserWarning, match='outside the histogram window'):
            probabilistic_constant_q(x, 2000.0, level_max_dB=middle, **kw)

    def test_a_window_holding_every_level_is_silent(self):
        x = self._noise()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            probabilistic_welch(x, 2000.0, segment_duration=0.5, nperseg=256)


class TestDecidecadeGuards:
    def _flat(self):
        f = np.linspace(50.0, 2000.0, 400)
        return np.ones_like(f), f

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_nonpositive_or_nonfinite_ref_raises(self, bad):
        p, f = self._flat()
        with pytest.raises(ConfigurationError,
                           match="ref must be > 0 Pa and finite"):
            decidecade_band_levels(p, frequencies=f, ref=bad)

    def test_negative_psd_raises_typed(self):
        # A negative-power band failed the `power > 0` publication test and
        # came back as a silent NaN level.
        p, f = self._flat()
        p[10] = -1.0
        with pytest.raises(ConfigurationError, match="negative"):
            decidecade_band_levels(p, frequencies=f)

    def test_flat_psd_gives_finite_levels_inside_support(self):
        p, f = self._flat()
        _, levels = decidecade_band_levels(p, frequencies=f)
        assert np.isfinite(levels).any()


class TestSilentZerosAreAnnounced:
    def test_sel_warns_for_bands_holding_no_fft_bin(self):
        # 'linear' bands are used as given, so a freq_max above Nyquist produces
        # whole bands that sum to exactly 0 Pa²·s — indistinguishable from a
        # measured silence. The octave ladders clamp instead.
        rng = np.random.default_rng(0)
        x = rng.normal(size=4000)
        with pytest.warns(UserWarning, match="no FFT bin"):
            r = _band_exposure(x, 2000.0, band_type='linear', freq_min=100.0, freq_max=3000.0,
                    n_bands=6)
        assert np.all(r.power[2:] == 0.0)
        # Bands entirely below Nyquist stay silent about it.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _band_exposure(x, 2000.0, band_type='linear', freq_min=100.0, freq_max=900.0,
                n_bands=4)

    @pytest.mark.parametrize(
        "estimator", ["welch",
                      "probabilistic_welch",
                      "spectrogram"])
    def test_complex_input_warns_about_the_two_sided_axis(self, estimator):
        # These three accept complex input and return an unsorted two-sided
        # axis, not the one-sided Pa²/Hz their docstrings describe.
        rng = np.random.default_rng(0)
        z = rng.normal(size=2048) + 1j * rng.normal(size=2048)
        with pytest.warns(UserWarning, match="TWO-SIDED"):
            if estimator == "welch":
                f = welch(z, 1000.0, nperseg=128).frequencies
            elif estimator == "probabilistic_welch":
                f = probabilistic_welch(z, 1000.0, segment_duration=0.5, nperseg=128).frequencies
            else:
                f = spectrogram(z, 1000.0, nperseg=128).frequencies
        assert f.min() < 0.0                       # the two-sided axis
        assert not np.all(np.diff(f) > 0)          # and it is not sorted

    @pytest.mark.parametrize("estimator",
                             [welch, spectrogram])
    def test_real_input_does_not_warn(self, estimator):
        x = np.random.default_rng(0).normal(size=2048)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimator(x, 1000.0, nperseg=128)


class TestPpsdOutOfWindowLevels:
    """When no PSD level lands inside ``[level_min_dB, level_max_dB]`` the pdf is
    all-NaN by construction; the histogram estimator suppresses numpy's
    per-column 0/0
    RuntimeWarnings and says so once, naming the window and the measured
    level span."""

    def _signal(self):
        return np.sin(2 * np.pi * 100.0 * np.arange(4000) / 1000.0) * 1e9

    def test_out_of_window_levels_warn_once_and_raise_no_runtime(self):
        with recorded_warnings() as caught:
            res = probabilistic_welch(self._signal(), 1000.0, segment_duration=0.5,
                       nperseg=256)
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        typed = [str(w.message) for w in caught
                 if 'histogram window' in str(w.message)]
        assert len(typed) == 1
        assert 'level_min_dB=0' in typed[0] and 'level_max_dB=150' in typed[0]
        assert 'dB re ref²' in typed[0]
        assert np.all(np.isnan(res.pdf))
        assert np.all(np.isfinite(res.mean_dB))

    def test_levels_partly_inside_the_window_skip_the_all_nan_notice(self):
        # The tone's bins land inside [100, 250]; the leakage floor far from
        # it does not, so those columns are named by the partial-loss
        # warning instead of the all-NaN one.
        with recorded_warnings() as caught:
            res = probabilistic_welch(self._signal(), 1000.0, segment_duration=0.5,
                       nperseg=256, level_min_dB=100, level_max_dB=250)
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        assert not [w for w in caught
                    if 'no PSD level falls inside' in str(w.message)]
        assert len([w for w in caught
                    if 'outside the histogram window' in str(w.message)]) == 1
        assert not np.all(np.isnan(res.pdf))


class TestDecidecadeSinglePositiveFrequencyGrid:
    """A grid whose only band support is one positive frequency (the
    two-sample rfftfreq grid) is reported in terms of the caller's
    ``frequencies`` argument, not the ``freq_min``/``freq_max`` arguments of the
    band helper it never called."""

    def test_error_names_the_callers_grid(self):
        with pytest.raises(ConfigurationError,
                           match='spans no decidecade band') as exc:
            decidecade_band_levels(np.ones(2), frequencies=np.fft.rfftfreq(2))
        assert 'freq_min' not in str(exc.value)

    def test_two_positive_frequencies_pass_the_guard(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            centers, levels = decidecade_band_levels(
                np.ones(3), frequencies=np.fft.rfftfreq(4, d=1e-3))
        assert centers.size > 0


def test_a_flattop_spectrum_reads_a_tone_between_bins():
    """Why :func:`welch` defaults to flattop rather than hann.

    The same half-bin-offset tone that costs 1.42 dB through hann (see
    :func:`test_psd_hann_worst_case_scalloping_is_1_42_dB`) costs about
    0.01 dB here, which is the difference between reading a tone's level and
    estimating it. The price is resolution: flattop's main lobe spans 3.77
    bins against hann's 1.50, so this default is wrong for separating
    neighbours and the caller says ``window='hann'`` when that is the job.
    """
    fs, nper = 1000.0, 256
    t = np.arange(8 * nper) / fs
    on = np.cos(2 * np.pi * (32 * fs / nper) * t)
    off = np.cos(2 * np.pi * (32.5 * fs / nper) * t)
    p_on = welch(on, fs, scaling='spectrum', nperseg=nper).power.max()
    p_off = welch(off, fs, scaling='spectrum', nperseg=nper).power.max()
    assert p_on == pytest.approx(0.5, rel=1e-3), "on-bin tone is not A**2/2"
    loss_dB = 10 * np.log10(p_off / p_on)
    assert abs(loss_dB) < 0.05, f"flattop scalloping is {loss_dB:.3f} dB"


def test_the_estimators_take_a_scaling_only_where_it_is_one_divide():
    """``scaling`` survives where the two answers are the same computation
    apart — divided, or not, by the bin's noise-equivalent bandwidth. It is
    absent where it would be a lie: an exposure is an energy, and no keyword
    turns it into a power.

    ``method`` is absent everywhere: the function IS the method, which is why
    nothing has to be policed against anything.
    """
    import inspect
    for estimator in (welch, constant_q, probabilistic_welch,
                      probabilistic_constant_q):
        taken = inspect.signature(estimator).parameters
        assert taken["scaling"].default == "density", estimator.__name__
        assert "method" not in taken, estimator.__name__
    for exposure in (sound_exposure, probabilistic_sound_exposure):
        taken = inspect.signature(exposure).parameters
        assert "scaling" not in taken, exposure.__name__
        assert "method" not in taken, exposure.__name__
    with pytest.raises(TypeError, match="scaling"):
        sound_exposure(np.ones(2048), 1000.0, scaling="density")
    with pytest.raises(ConfigurationError, match="unknown scaling"):
        welch(np.ones(2048), 1000.0, scaling="rms")


def test_each_function_carries_the_window_its_statistic_measures_best():
    """The two Welch doors differ in exactly the two things the split is
    about: what they normalise by, and the window that suits it."""
    x = np.random.default_rng(5).standard_normal(4096)
    fs = 8000.0
    density = welch(x, fs, nperseg=256)
    spectrum = welch(x, fs, scaling='spectrum', nperseg=256)
    assert density.scaling == "density" and spectrum.scaling == "spectrum"
    np.testing.assert_allclose(
        density.power,
        welch(x, fs, nperseg=256, window="hann").power)
    np.testing.assert_allclose(
        spectrum.power,
        welch(x, fs, scaling='spectrum', nperseg=256, window="flattop").power)


def test_an_exposure_takes_none_of_the_knobs_that_would_break_it():
    """The guard that used to police these is now the signature itself: a
    band's value is the sum of the bins inside it, which is the band's energy
    only when every bin is counted once and whole, so there is no window, no
    overlap, no detrending and no averaging to set."""
    import inspect
    taken = inspect.signature(sound_exposure).parameters
    for absent in ("window", "noverlap", "detrend", "average", "nfft"):
        assert absent not in taken, absent
    for knob in ("window", "noverlap", "detrend", "average"):
        with pytest.raises(TypeError, match=knob):
            sound_exposure(np.ones(4096), 1000.0, **{knob: "hann"})


def test_a_transient_moves_the_mean_estimate_and_not_the_median():
    """``average='median'`` is forwarded because it changes the estimate.

    An ambient record with a passing ship in it is the case the median
    estimator exists for: a few loud segments pull the mean of the
    periodograms and leave the median where the background is.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200000)
    x[5000:5100] += 50.0                       # one short, loud transient
    mean = welch(x, 48000.0, nperseg=1024).power
    median = welch(x, 48000.0, nperseg=1024,
                                    average="median").power
    lift_dB = 10 * np.log10(mean.mean() / median.mean())
    assert lift_dB > 1.0, (
        f"the transient moved the two estimators by only {lift_dB:.2f} dB; "
        f"average= is not reaching scipy")


def test_detrend_false_keeps_the_dc_bin_scipy_removes():
    """scipy detrends each segment's constant by default, which empties the
    DC bin; ``detrend=False`` is how a caller keeps it."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(60000) + 3.0       # a deliberate offset
    removed = welch(x, 8000.0, nperseg=1024).power[0]
    kept = welch(x, 8000.0, nperseg=1024,
                                  detrend=False).power[0]
    assert kept > removed * 10, (
        f"DC bin {kept:.3e} against {removed:.3e}: detrend= is not reaching "
        f"scipy")


@pytest.mark.parametrize('estimator', ['welch', 'spectrogram',
                                       'probabilistic_welch'])
def test_every_segmenting_estimator_takes_welchs_detrend(estimator):
    """Same keyword, same default, same effect on welch, spectrogram and
    probabilistic_welch. On a transient the default fabricates low-frequency
    energy: a Gaussian-windowed 200 Hz tone's segments have a mean the pulse
    does not have, and subtracting it lifts the DC bin by tens of dB."""
    import inspect
    fs = 8000.0
    t = np.arange(16000) / fs
    pulse = np.exp(-((t - 1.0) / 0.02) ** 2) * np.cos(2 * np.pi * 200 * t)
    fn = {'welch': welch, 'spectrogram': spectrogram,
          'probabilistic_welch': probabilistic_welch}[estimator]
    assert inspect.signature(fn).parameters['detrend'].default == 'constant'

    def dc(detrend):
        if estimator == 'welch':
            return welch(pulse, fs, nperseg=1024, detrend=detrend).power[0]
        if estimator == 'spectrogram':
            return spectrogram(pulse, fs, nperseg=1024,
                               detrend=detrend).power[0].max()
        est = probabilistic_welch(pulse, fs, segment_duration=0.5, nperseg=1024,
                                  detrend=detrend, level_min_dB=-400, level_max_dB=100)
        return 10 ** (np.nanmax(np.asarray(est.mean_dB)[0]) / 10)
    lift_dB = 10 * np.log10(dc('constant') / dc(False))
    assert lift_dB > 20.0, f"default lifts DC by only {lift_dB:.1f} dB"


def test_an_argument_no_estimator_takes_is_refused_by_python_itself():
    """No estimator forwards a ``**options`` bag, so a misplaced argument is
    a plain ``TypeError`` naming it rather than a value silently ignored."""
    with pytest.raises(TypeError, match="return_onesided"):
        welch(np.ones(2048), 1000.0, return_onesided=False)
    with pytest.raises(TypeError, match="bins_per_octave"):
        welch(np.ones(2048), 1000.0, scaling='spectrum', bins_per_octave=12)
    with pytest.raises(TypeError, match="nperseg"):
        constant_q(np.ones(2048), 1000.0, scaling='spectrum', nperseg=512)


def _band_weights(frequencies, bands, fs):
    """Share of each bin every band takes, the way the estimator assigns them:
    bin k>0 is the interval ``f ± df/2`` (Nyquist its lower half), split
    between the bands it overlaps; DC is in none."""
    f = np.asarray(frequencies)
    return [_overlap_weights(f, f[1] - f[0], fs, lo, hi)
            for lo, _c, hi in bands]


class TestEveryEstimatorAnswersAndSaysWhatItReturned:
    """One function per statistic, each carrying what it produced.

    There is no ``method`` or ``scaling`` to cross any more — the name at the
    call site is the choice — but the result still says which statistic it
    holds, because a plot axis reading "/Hz" over band power is wrong by the
    window's noise-equivalent bandwidth and says nothing about being wrong.
    """

    FS = 8000.0

    def _signal(self):
        return np.random.default_rng(4).standard_normal(int(4 * self.FS))

    #: Every averaged estimator, with the arguments it needs and what it
    #: should report having computed.
    AVERAGED = [
        (welch, dict(nperseg=1024), 'density', 'welch'),
        (welch, dict(nperseg=1024, scaling='spectrum'), 'spectrum', 'welch'),
        (sound_exposure, dict(freq_min=10.0, freq_max=2000.0), 'exposure', 'welch'),
        (constant_q, dict(freq_min=50.0, freq_max=2000.0, scaling='spectrum'),
         'spectrum', 'constant_q'),
        (constant_q, dict(freq_min=50.0, freq_max=2000.0), 'density', 'constant_q'),
    ]

    @pytest.mark.parametrize('door,options,scaling,method', AVERAGED,
                             ids=[row[0].__name__ for row in AVERAGED])
    def test_an_averaged_estimate_carries_its_own_statistic(
            self, door, options, scaling, method):
        est = door(self._signal(), self.FS, **options)
        assert (est.scaling, est.method) == (scaling, method)
        assert est.power.shape == est.frequencies.shape
        assert np.all(est.power[np.isfinite(est.power)] >= 0.0)
        # Only the exposure sits on standard bands, and it carries their
        # edges and the ladder's name for the bar plot.
        banded = door is sound_exposure
        assert (est.bands is not None) == banded
        assert (est.band_type == 'decidecade') == banded

    PROBABILISTIC = [
        (probabilistic_welch, dict(nperseg=1024), 'density', 'welch'),
        (probabilistic_welch, dict(nperseg=1024, scaling='spectrum'),
         'spectrum', 'welch'),
        (probabilistic_sound_exposure, dict(freq_min=10.0, freq_max=2000.0),
         'exposure', 'welch'),
        (probabilistic_constant_q,
         dict(freq_min=50.0, freq_max=2000.0, scaling='spectrum'), 'spectrum',
         'constant_q'),
        (probabilistic_constant_q, dict(freq_min=50.0, freq_max=2000.0), 'density',
         'constant_q'),
    ]

    @pytest.mark.parametrize('door,options,scaling,method', PROBABILISTIC,
                             ids=[row[0].__name__ for row in PROBABILISTIC])
    def test_a_histogram_carries_its_own_statistic_too(self, door, options,
                                                       scaling, method):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est = door(self._signal(), self.FS, **options)
        assert (est.scaling, est.method) == (scaling, method)
        assert est.pdf.shape == (est.level_edges.size - 1,
                                 est.frequencies.size)
        assert est.mean_dB.shape == est.frequencies.shape
        assert est.ref == pytest.approx(1e-6)

    def test_a_band_exposure_is_the_records_own_energy_in_those_bands(self):
        """Parseval, which is what makes an exposure a measurement rather
        than a scaled spectrum: boxcar, no overlap, no detrending, and the
        record padded to whole segments so nothing is dropped.

        The total is the energy over the SPAN the bands cover, not the whole
        record's: every band edge is > 0, so the DC bin belongs to no band.
        That is 0.08 % of a white record here, and it is the difference
        between a total that is exact and one that is nearly right."""
        x = self._signal()
        nperseg = 1024
        est = sound_exposure(x, self.FS, band_type='linear', freq_min=1e-9,
                             freq_max=self.FS / 2, n_bands=64, nperseg=nperseg)
        n_seg = int(np.ceil(x.size / nperseg))
        padded = np.pad(x, (0, n_seg * nperseg - x.size))
        per_bin = welch(padded, self.FS, scaling='spectrum', nperseg=nperseg,
                                 window='boxcar', noverlap=0, detrend=False)
        duration = n_seg * nperseg / self.FS
        above_dc = per_bin.frequencies > 0.0
        assert est.power.sum() == pytest.approx(
            per_bin.power[above_dc].sum() * duration, rel=1e-12)
        # and that span is all but the DC bin of the record's own energy
        assert est.power.sum() == pytest.approx(np.sum(x ** 2) / self.FS,
                                                rel=1e-2)

    def test_an_exposure_is_the_band_power_times_the_duration(self):
        """The relation the statistic is defined by, band for band."""
        x = self._signal()
        n_seg, nperseg = 32, 1024
        padded = np.pad(x, (0, n_seg * nperseg - x.size))
        power = welch(padded, self.FS, scaling='spectrum', nperseg=nperseg, noverlap=0,
                               detrend=False, window='boxcar')
        exposure = sound_exposure(x, self.FS, band_type='linear', freq_min=1e-9,
                                  freq_max=self.FS / 2, n_bands=64,
                                  nperseg=nperseg)
        duration = n_seg * nperseg / self.FS
        by_hand = np.array([power.power @ w * duration
                            for w in _band_weights(power.frequencies,
                                                   exposure.bands, self.FS)])
        np.testing.assert_allclose(exposure.power, by_hand, rtol=1e-12)
        assert by_hand.sum() > 0.9 * np.sum(x ** 2) / self.FS

    def test_a_bin_estimate_cannot_be_drawn_as_bars(self):
        est = welch(self._signal(), self.FS, scaling='spectrum', nperseg=512)
        with pytest.raises(ConfigurationError, match='no band_type'):
            plot_sel(est)

    def test_an_unknown_ladder_names_the_ones_that_exist(self):
        with pytest.raises(ConfigurationError, match='unknown band_type'):
            sound_exposure(self._signal(), self.FS, band_type='half_octave')


class TestEachEstimatorBringsItsOwnDefaults:
    """A caller asks for the statistic, not for the settings it needs.

    The window, the overlap and the detrending differ by statistic, because
    the three measure different things: a density averages noise, a spectrum
    reads a tone's level, an exposure counts the energy that passed the
    sensor. Each function's defaults are the ones its own statistic needs, so
    the name at the call site is the whole configuration.
    """

    FS = 8000.0

    def _tone_half_a_bin_off(self, nperseg=1024):
        # Worst case for scalloping: a tone exactly between two bins.
        f0 = (10.5 * self.FS) / nperseg
        t = np.arange(int(8 * self.FS)) / self.FS
        return np.sqrt(2.0) * np.sin(2 * np.pi * f0 * t), f0

    def test_a_spectrum_reads_an_off_bin_tone_right_by_default(self):
        """Flat-top by default, so a tone between two bins reads right to a
        hundredth of a dB instead of hann's 1.42 dB low."""
        x, _ = self._tone_half_a_bin_off()
        flattop = welch(x, self.FS, scaling='spectrum', nperseg=1024).power.max()
        hann = welch(x, self.FS, scaling='spectrum', nperseg=1024,
                              window='hann').power.max()
        assert 10 * np.log10(flattop) == pytest.approx(0.0, abs=0.05)
        assert 10 * np.log10(hann) == pytest.approx(-1.42, abs=0.1)

    def test_a_density_is_the_same_under_either_window_by_default(self):
        """Which is what a density is FOR: the noise-equivalent bandwidth
        divides out, so the default costs nothing and a caller who overrides
        it gets the same answer."""
        x = np.random.default_rng(3).standard_normal(int(8 * self.FS))
        default = welch(x, self.FS, nperseg=1024).power
        boxcar = welch(x, self.FS, nperseg=1024,
                                        window='boxcar').power
        assert np.median(default) == pytest.approx(np.median(boxcar),
                                                   rel=0.05)

    def test_an_exposure_runs_on_the_settings_parseval_needs(self):
        """It takes none of them as arguments, so this checks the estimate
        equals the Welch one computed with all four spelled out."""
        x = np.random.default_rng(5).standard_normal(int(4 * self.FS))
        nperseg = 1024
        n_seg = int(np.ceil(x.size / nperseg))
        padded = np.pad(x, (0, n_seg * nperseg - x.size))
        spelled = welch(padded, self.FS, scaling='spectrum', nperseg=nperseg,
                                 window='boxcar', noverlap=0, detrend=False)
        duration = n_seg * nperseg / self.FS
        est = sound_exposure(x, self.FS, band_type='linear', freq_min=1e-9,
                             freq_max=self.FS / 2, n_bands=32, nperseg=nperseg)
        by_hand = np.array([spelled.power @ w * duration
                            for w in _band_weights(spelled.frequencies,
                                                   est.bands, self.FS)])
        np.testing.assert_allclose(est.power, by_hand, rtol=1e-12)

    def test_a_median_average_is_the_robust_choice_for_a_density(self):
        """The knob is offered where it is a level, and absent where it would
        be an energy: one loud quarter of a tape moves the mean of the
        periodograms and leaves the median on the background."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(int(8 * self.FS))
        x[-x.size // 4:] *= 10.0
        mean = welch(x, self.FS, nperseg=1024).power
        median = welch(x, self.FS, nperseg=1024,
                                        average='median').power
        assert np.median(10 * np.log10(mean / median)) > 10.0
        assert np.median(10 * np.log10(median)) == pytest.approx(
            10 * np.log10(2.0 / self.FS), abs=2.5)

    @pytest.mark.parametrize('door,unit,options', [
        (welch, 'Pa²/Hz', dict(nperseg=512)),
        (welch, 'Pa²', dict(nperseg=512, scaling='spectrum')),
        (sound_exposure, 'Pa²·s', dict(freq_min=10.0, freq_max=2000.0)),
    ], ids=['density', 'spectrum', 'exposure'])
    def test_the_level_axis_names_the_unit_the_estimator_produced(
            self, door, unit, options):
        x = np.random.default_rng(9).standard_normal(int(2 * self.FS))
        fig, ax = door(x, self.FS, **options).plot()
        assert ax.get_ylabel().endswith(f"{unit})")
        plt.close(fig)


class TestAnExposureIsTheWelchBinsSummed:
    """``sound_exposure`` is not a second estimator: it calls the Welch route
    per batch and sums the bins each band covers, so there is one
    implementation underneath. These pin the arithmetic on top of it — the
    banding, and the batching, which must change memory and not the answer.
    """

    FS = 2000.0

    def _record(self, seconds=8):
        return np.random.default_rng(11).standard_normal(int(seconds * self.FS))

    def _integrate(self, welch, bands, duration):
        f = np.asarray(welch.frequencies)
        df = f[1] - f[0]
        return np.array([welch.power @ _overlap_weights(f, df, self.FS, lo, hi)
                         * duration
                         for lo, _c, hi in bands])

    @pytest.mark.parametrize('nperseg', [500, 2000])
    def test_a_band_exposure_is_the_welch_bins_it_covers(self, nperseg):
        x = self._record()
        band = sound_exposure(x, self.FS, freq_min=10.0, freq_max=900.0,
                              nperseg=nperseg)
        n_seg = int(np.ceil(x.size / nperseg))
        padded = np.pad(x, (0, n_seg * nperseg - x.size))
        per_bin = welch(padded, self.FS, scaling='spectrum',
                        window='boxcar', nperseg=nperseg, noverlap=0,
                        detrend=False)
        np.testing.assert_allclose(
            band.power,
            self._integrate(per_bin, band.bands, n_seg * nperseg / self.FS),
            rtol=1e-12)

    def test_a_flat_spectrum_reads_the_same_density_in_every_band(self):
        """Each band integrates exactly its own width. A record whose per-bin
        exposure is exactly flat therefore gives the same exposure per hertz
        in every decidecade band; counting whole bins by centre read the
        10 Hz band (3 bins in 2.31 Hz) +1.14 dB high and the 16 Hz band
        (3 bins in 3.66 Hz) -0.86 dB low."""
        fs = 1000.0
        rng = np.random.default_rng(3)
        spectrum = np.exp(2j * np.pi * rng.random(int(fs) // 2 + 1))
        spectrum[0] = spectrum[-1] = 1.0
        x = np.tile(np.fft.irfft(spectrum, n=int(fs)), 4)
        out = sound_exposure(x, fs, freq_min=8.9125, freq_max=400.0)
        widths = out.bands[:, 2] - out.bands[:, 0]
        density_dB = 10 * np.log10(out.power / widths)
        assert np.ptp(density_dB) < 1e-9

    def test_batching_changes_nothing_when_batches_hold_whole_segments(self):
        """The batching is a memory strategy, not an estimator choice: with
        ``batch_size`` a multiple of ``nperseg`` every batch ends on a
        segment boundary, so the answer is the unbatched one."""
        x = self._record(seconds=16)
        nperseg = 1000
        whole = sound_exposure(x, self.FS, freq_min=10.0, freq_max=900.0,
                               nperseg=nperseg)
        batched = sound_exposure(x, self.FS, freq_min=10.0, freq_max=900.0,
                                 nperseg=nperseg, batch_size=4 * nperseg)
        np.testing.assert_allclose(batched.power, whole.power, rtol=1e-12)

    def test_the_default_batch_holds_whole_segments_and_stays_silent(self):
        """A default that warned on every call is a default that is wrong:
        a fixed sample count never divides ``nperseg``, which defaults to the
        sample rate, so the default is written in segments instead."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            sound_exposure(self._record(seconds=600), self.FS, freq_min=10.0,
                           freq_max=900.0)

    def test_a_batch_that_splits_a_segment_says_so(self):
        x = self._record(seconds=16)
        with pytest.warns(UserWarning, match='multiple of nperseg'):
            sound_exposure(x, self.FS, freq_min=10.0, freq_max=900.0, nperseg=1000,
                           batch_size=1500)


class TestEveryEstimatorSpellsTheRangeAndTheRecordTheSameWay:
    """``freq_min`` / ``freq_max`` / ``integration_time`` mean the same thing in every
    estimator: the frequency range of the estimate, and the stretch of record
    it is taken over.

    They are the arguments that survived the split, because one sentence
    describes each of them everywhere. What an unset one falls back to is the
    estimator's own answer — Welch resolves the whole spectrum, constant-Q
    starts at 20 Hz, the ladder spans the reporting range — which is why they
    are not a shared default.
    """

    FS = 2000.0

    def _record(self, seconds=8):
        return np.random.default_rng(13).standard_normal(int(seconds * self.FS))

    DOORS = [(welch, dict(nperseg=512)),
             (welch, dict(nperseg=512)),
             (constant_q, {}),
             (sound_exposure, {})]

    @pytest.mark.parametrize('door,options', DOORS,
                             ids=[row[0].__name__ for row in DOORS])
    def test_the_range_bounds_the_axis(self, door, options):
        est = door(self._record(), self.FS, freq_min=50.0, freq_max=500.0, **options)
        assert est.frequencies[0] >= 50.0
        assert est.frequencies[-1] <= 500.0
        assert est.frequencies.size > 1

    @pytest.mark.parametrize('door,options', DOORS,
                             ids=[row[0].__name__ for row in DOORS])
    def test_integration_time_takes_the_first_seconds_of_the_record(
            self, door, options):
        x = self._record()
        n = int(2.0 * self.FS)
        trimmed = door(x, self.FS, integration_time=2.0, **options)
        by_hand = door(x[:n], self.FS, **options)
        np.testing.assert_allclose(trimmed.power, by_hand.power, rtol=1e-12)

    def test_an_integration_time_shorter_than_a_sample_says_so(self):
        with pytest.raises(ConfigurationError,
                           match="no samples to integrate"):
            welch(self._record(), self.FS,
                                   integration_time=1e-9)

    def test_a_histogram_takes_the_same_three(self):
        x = self._record(seconds=16)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = probabilistic_sound_exposure(x, self.FS, segment_duration=1.0,
                                             freq_min=50.0, freq_max=500.0,
                                             integration_time=8.0)
        assert r.frequencies[0] >= 50.0 and r.frequencies[-1] <= 500.0
        assert r.bands is not None


class TestTheExposureHistogramSummarisesPerSegmentEnergy:
    """``probabilistic_sound_exposure`` is how a monitoring record reports what
    a day of piling delivered rather than what its loudest minute did: each
    sample is ONE segment's energy.

    That makes it the one estimator whose level axis moves with an argument —
    doubling ``segment_duration`` doubles the energy each sample integrates — which
    is a property to pin rather than a surprise to discover on a report.
    """

    FS = 2000.0

    def _record(self, seconds=32):
        return np.random.default_rng(19).standard_normal(int(seconds * self.FS))

    def _run(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return probabilistic_sound_exposure(
                self._record(), self.FS, freq_min=10.0, freq_max=900.0, **kwargs)

    def test_it_reports_what_it_computed(self):
        r = self._run(segment_duration=1.0)
        assert (r.scaling, r.method, r.band_type) == ('exposure', 'welch',
                                                      'decidecade')
        assert r.bands is not None and len(r.bands) == r.frequencies.size
        assert r.segment_duration == 1.0
        assert r.pdf.shape == (r.level_edges.size - 1, r.frequencies.size)

    def test_doubling_the_segment_adds_three_dB_to_every_sample(self):
        """Each sample is that segment's energy, so twice the segment is twice
        the energy: +3.01 dB, band for band, and nothing else moves."""
        one = self._run(segment_duration=1.0, level_min_dB=-60, level_max_dB=60)
        two = self._run(segment_duration=2.0, level_min_dB=-60, level_max_dB=60)
        shift = two.mean_dB - one.mean_dB
        assert np.nanmedian(shift) == pytest.approx(10 * np.log10(2.0),
                                                    abs=0.15)

    def test_its_mean_tracks_the_whole_record_exposure_per_segment(self):
        """The histogram and the averaged estimator describe the same record:
        a segment's exposure is the record's divided by the number of
        segments, so the two differ by that count and not by anything else."""
        x = self._record()
        seg = 4.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hist = probabilistic_sound_exposure(x, self.FS, segment_duration=seg,
                                                segment_overlap_percent=0, freq_min=10.0,
                                                freq_max=900.0, level_min_dB=-60,
                                                level_max_dB=60)
            whole = sound_exposure(x, self.FS, freq_min=10.0, freq_max=900.0)
        n_segments = x.size / (seg * self.FS)
        expected = power_to_dB(whole.power / n_segments, 1e-6)
        assert np.nanmedian(hist.mean_dB - expected) == pytest.approx(0.0,
                                                                     abs=0.6)


class TestAnEnergyIsNotAScalingOfABinEstimator:
    """``scaling='exposure'`` belongs to :func:`sound_exposure` alone.

    The private core still knows the word — the band route computes an
    exposure through it — but no bin estimator may be told it, because there
    the window and the overlap are the caller's and an energy needs neither.
    Reached through ``welch`` at its own defaults it returned 1.0088x the
    record's energy (50 % overlap inventing 0.88 %), and 1.537x under
    ``window='hann'``, silently: the one spelling no test exercised.
    """

    FS = 8000.0

    def _record(self):
        return np.random.default_rng(23).standard_normal(int(4 * self.FS))

    @pytest.mark.parametrize('estimator,options', [
        (welch, {}), (constant_q, dict(freq_max=900.0)),
        (probabilistic_welch, {}),
        (probabilistic_constant_q, dict(freq_max=900.0)),
    ], ids=['welch', 'constant_q', 'probabilistic_welch',
            'probabilistic_constant_q'])
    def test_a_bin_estimator_refuses_the_exposure_scaling(self, estimator,
                                                          options):
        with pytest.raises(ConfigurationError,
                           match="unknown scaling 'exposure'") as excinfo:
            estimator(self._record(), self.FS, scaling='exposure', **options)
        message = str(excinfo.value)
        # and it names the function that does compute one
        assert 'sound_exposure' in message
        assert "'density'" in message and "'spectrum'" in message

    def test_the_remediation_does_not_advertise_the_third_value(self):
        """A caller who mistypes a scaling used to be told to use
        ``'exposure'`` — the one value that silently returned a wrong
        number."""
        with pytest.raises(ConfigurationError,
                           match="unknown scaling 'rms'") as excinfo:
            welch(self._record(), self.FS, scaling='rms')
        assert "'exposure'" not in str(excinfo.value)

    def test_the_band_route_reaches_it_and_is_exact(self):
        """The word stays live inside the package: this is the estimate that
        uses it, and it is Parseval-exact, which is what the refusal
        protects."""
        x = self._record()
        est = sound_exposure(x, self.FS, band_type='linear', freq_min=1e-9,
                             freq_max=self.FS / 2, n_bands=32, nperseg=1024)
        assert est.scaling == 'exposure'
        assert est.power.sum() == pytest.approx(np.sum(x ** 2) / self.FS,
                                                rel=1e-2)


# ── Multichannel records: the time axis is named, never guessed wrong ───────
# read_wav returns (n_samples, n_channels); along the last axis welch made
# 48 000 two-bin "spectra" and sound_exposure a plausible wrong total.

class TestMultichannelTimeAxis:
    FS = 8000.0

    def _record(self):
        rng = np.random.default_rng(3)
        return rng.standard_normal((16000, 3))          # read_wav's layout

    def test_the_wav_layout_is_refused_without_axis(self):
        from uacpy.acoustic_signal import welch, spectrogram
        from uacpy.core.exceptions import ConfigurationError
        x = self._record()
        with pytest.raises(ConfigurationError, match="axis=0"):
            welch(x, self.FS, nperseg=1024)
        with pytest.raises(ConfigurationError, match="axis=0"):
            spectrogram(x, self.FS, nperseg=256)

    def test_axis_zero_is_one_spectrum_per_channel(self):
        from uacpy.acoustic_signal import welch
        x = self._record()
        est = welch(x, self.FS, nperseg=1024, axis=0)
        assert est.power.shape == (3, est.frequencies.size)
        for ch in range(3):
            np.testing.assert_allclose(
                est.power[ch], welch(x[:, ch], self.FS, nperseg=1024).power,
                rtol=1e-12)

    def test_sound_exposure_takes_one_channel(self):
        from uacpy.acoustic_signal import sound_exposure
        from uacpy.core.exceptions import ConfigurationError
        x = self._record()
        for arr in (x, x.T):
            with pytest.raises(ConfigurationError, match="one channel"):
                sound_exposure(arr, self.FS)

    def test_spectrogram_axis_puts_time_last(self):
        from uacpy.acoustic_signal import spectrogram
        x = self._record()
        res = spectrogram(x, self.FS, nperseg=256, axis=0)
        one = spectrogram(x[:, 1], self.FS, nperseg=256)
        np.testing.assert_allclose(res.power[1], one.power, rtol=1e-12)

    def test_a_nested_list_takes_the_same_axis_rule_as_its_array(self):
        from uacpy.acoustic_signal import constant_q, welch
        from uacpy.core.exceptions import ConfigurationError
        x = self._record()
        as_list = x.tolist()
        np.testing.assert_array_equal(
            welch(as_list, self.FS, nperseg=1024, axis=0).power,
            welch(x, self.FS, nperseg=1024, axis=0).power)
        kw = dict(freq_min=100.0, freq_max=2000.0, bins_per_octave=12)
        np.testing.assert_array_equal(
            constant_q(as_list[:4000], self.FS, axis=0, **kw).power,
            constant_q(x[:4000], self.FS, axis=0, **kw).power)
        with pytest.raises(ConfigurationError, match="axis=0"):
            welch(as_list, self.FS, nperseg=1024)

    def test_a_plain_list_is_one_record_on_the_histogram_too(self):
        from uacpy.acoustic_signal import probabilistic_welch
        x = self._record()[:, 0]
        np.testing.assert_array_equal(
            probabilistic_welch(list(x), self.FS, nperseg=1024).pdf,
            probabilistic_welch(x, self.FS, nperseg=1024).pdf)

    def test_probabilistic_constant_q_pools_every_channel(self):
        from uacpy.acoustic_signal import probabilistic_constant_q
        x = self._record()
        kw = dict(freq_min=200.0, freq_max=2000.0, bins_per_octave=6, level_step_dB=0.5,
                  level_min_dB=0.0, level_max_dB=200.0)
        pooled = probabilistic_constant_q(x, self.FS, axis=0, **kw)
        records = probabilistic_constant_q(list(x.T), self.FS, **kw)
        np.testing.assert_array_equal(pooled.pdf, records.pdf)
        # The pooled mean level is the mean over every channel's frames,
        # which for equal-length channels is the mean of the per-channel
        # means.
        one = [probabilistic_constant_q(x[:, ch], self.FS, **kw).mean_dB
               for ch in range(3)]
        np.testing.assert_allclose(pooled.mean_dB, np.mean(one, axis=0),
                                   rtol=1e-12)


class TestSpectralResultsKeepWhatTheyMean:
    def test_replace_keeps_the_scaling(self):
        from uacpy.acoustic_signal import welch
        x = np.random.default_rng(1).standard_normal(8192)
        est = welch(x, 1000.0, nperseg=512, scaling='spectrum')
        moved = est._replace(power=est.power * 2.0)
        assert moved.scaling == 'spectrum' and moved.method == 'welch'
        with pytest.raises(TypeError,
                           match=r"unexpected field names: \['bogus'\]"):
            est._replace(bogus=1)

    def test_spectrogram_records_and_plots_its_scaling(self):
        from uacpy.acoustic_signal import spectrogram
        x = np.random.default_rng(2).standard_normal(8192)
        res = spectrogram(x, 1000.0, nperseg=256, scaling='spectrum')
        assert res.scaling == 'spectrum' and res.mode == 'psd'
        assert res._replace(power=res.power).scaling == 'spectrum'
        fig, ax = res.plot()
        try:
            label = fig.axes[-1].get_ylabel()
            assert label.endswith("Pa²)") and "/Hz" not in label
        finally:
            plt.close(fig)

    def test_a_phase_panel_is_not_drawn_as_a_level(self):
        from uacpy.acoustic_signal import spectrogram
        from uacpy.core.exceptions import ConfigurationError
        x = np.random.default_rng(2).standard_normal(8192)
        res = spectrogram(x, 1000.0, nperseg=256, mode='angle')
        with pytest.raises(ConfigurationError, match="phase"):
            res.plot()


# ── one Welch default overlap, set by the window (Abraham §9.2.10.1) ──────


def test_the_default_overlap_keeps_every_independent_sample_of_the_window():
    """Half a segment under hann and 78.3 % (rounded up) under flat-top, whose
    independent sample rate eta_w = 4.60 allows a spacing of N/4.60 at most;
    other windows keep half a segment."""
    from uacpy.acoustic_signal.windows import _default_noverlap
    assert _default_noverlap('hann', 1024) == 512
    assert _default_noverlap('hann', 1025) == 512
    assert _default_noverlap('flattop', 1024) == 802       # ceil(0.7826 * 1024)
    assert _default_noverlap('FLATTOP', 100) == 79
    assert _default_noverlap('boxcar', 1024) == 512
    assert _default_noverlap(np.ones(64), 64) == 32
    assert _default_noverlap('flattop', 1) == 0


def test_the_flattop_overlap_is_abrahams_rule_on_the_window_itself():
    """eta_w by Abraham's Eq. 4.122, N (sum w^2)^2 / sum R_w^2, on scipy's
    flattop: the constant is that window's, not a rounded figure."""
    from scipy.signal import get_window
    from uacpy.acoustic_signal.windows import _FLATTOP_OVERLAP
    w = get_window('flattop', 4096)
    r = np.correlate(w, w, mode='full')
    eta = w.size * np.sum(w ** 2) ** 2 / np.sum(r ** 2)
    assert 1.0 - 1.0 / eta == pytest.approx(_FLATTOP_OVERLAP, abs=1e-3)


@pytest.mark.parametrize('scaling, window, noverlap', [
    ('density', 'hann', 512), ('spectrum', 'flattop', 802)])
def test_welch_segments_with_the_default_overlap_of_its_window(
        scaling, window, noverlap):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(20000)
    got = welch(x, 1000.0, scaling=scaling, nperseg=1024)
    _, want = _scipy_welch(x, 1000.0, window=window, nperseg=1024,
                           noverlap=noverlap, scaling=scaling)
    np.testing.assert_array_equal(np.asarray(got.power), want)


def test_probabilistic_welch_segments_with_the_window_default_overlap(
        monkeypatch):
    """The histogram's per-segment Welch takes the same window rule as welch:
    flat-top at 78.3 % for a spectrum, hann at half for a density."""
    from uacpy.acoustic_signal import spectral
    seen = []
    scipy_welch = spectral._sig.welch

    def spy(*args, **kwargs):
        seen.append((kwargs['window'], kwargs['noverlap']))
        return scipy_welch(*args, **kwargs)

    monkeypatch.setattr(spectral._sig, 'welch', spy)
    x = np.random.default_rng(5).standard_normal(8000)
    probabilistic_welch(x, 1000.0, segment_duration=2.0, nperseg=1024,
                        scaling='spectrum')
    probabilistic_welch(x, 1000.0, segment_duration=2.0, nperseg=1024)
    assert set(seen) == {('flattop', 802), ('hann', 512)}


def test_the_frf_welch_default_overlaps_like_welch():
    """``frf_welch`` overlaps as welch does: half a segment under the
    default Hann window, the flat-top overlap under ``window='flattop'``,
    with ``FRF`` passing its window through."""
    from uacpy.acoustic_signal.frf import FRF
    rng = np.random.default_rng(4)
    x = rng.standard_normal(20000)
    y = np.convolve(x, [0.5, 0.3], mode='same')
    coherence = FRF(nperseg=1024).compute(x, y, 1000.0).coherence
    _, pxx = _scipy_welch(x, 1000.0, nperseg=1024, noverlap=512)
    _, pyy = _scipy_welch(y, 1000.0, nperseg=1024, noverlap=512)
    from scipy.signal import csd
    _, pxy = csd(x, y, 1000.0, nperseg=1024, noverlap=512)
    np.testing.assert_allclose(np.abs(coherence), np.abs(pxy) ** 2 / (pxx * pyy),
                               rtol=1e-10)
    flat = FRF(nperseg=1024, window='flattop').compute(x, y, 1000.0)
    options = dict(nperseg=1024, noverlap=802, window='flattop')
    _, pxx = _scipy_welch(x, 1000.0, **options)
    _, pyy = _scipy_welch(y, 1000.0, **options)
    _, pxy = csd(x, y, 1000.0, **options)
    np.testing.assert_allclose(np.abs(flat.coherence),
                               np.abs(pxy) ** 2 / (pxx * pyy), rtol=1e-10)


class TestAHistogramKeepsItsSegments:
    """The per-segment levels a ``probabilistic_*`` estimate is built from
    are kept, and rebuild its pdf, mean and spread exactly."""

    FS = 2000.0

    def _noise(self, seconds=6.0, seed=3):
        return (np.random.default_rng(seed).standard_normal(
            int(seconds * self.FS)) * 3e-3)

    @pytest.mark.parametrize('estimator, kw', [
        ('probabilistic_welch', {'segment_duration': 1.0, 'nperseg': 256}),
        ('probabilistic_sound_exposure', {'segment_duration': 1.0})])
    def test_the_segments_rebuild_the_histogram_bit_exactly(self, estimator,
                                                            kw):
        from uacpy import acoustic_signal as S
        r = getattr(S, estimator)(self._noise(), self.FS, **kw)
        pdf, mean, std = S.level_histogram(r.segment_levels_dB, r.level_edges)
        assert np.array_equal(pdf, r.pdf, equal_nan=True)
        assert np.array_equal(mean, r.mean_dB)
        assert np.array_equal(std, r.std_dB)
        assert r.segment_levels_dB.shape == (r.segment_times.size,
                                             r.frequencies.size)

    def test_segment_times_are_centres_restarting_per_record(self):
        from uacpy import acoustic_signal as S
        x = self._noise(3.0)
        one = S.probabilistic_welch(x, self.FS, segment_duration=1.0,
                                    segment_overlap_percent=50, nperseg=256)
        assert np.allclose(one.segment_times, [0.5, 1.0, 1.5, 2.0, 2.5])
        two = S.probabilistic_welch(np.stack([x, x], axis=1), self.FS,
                                    segment_duration=1.0, segment_overlap_percent=50,
                                    nperseg=256, axis=0)
        assert np.array_equal(two.segment_times,
                              np.concatenate([one.segment_times] * 2))

    def test_percentiles_are_over_every_segment(self):
        from uacpy import acoustic_signal as S
        r = S.probabilistic_welch(self._noise(), self.FS, segment_duration=0.5,
                                  nperseg=256, level_min_dB=38, level_max_dB=40)
        got = r.percentiles([5, 50, 95])
        assert np.array_equal(got, np.percentile(r.segment_levels_dB,
                                                 [5, 50, 95], axis=0))
        # The window clips pdf, not the percentiles.
        assert got.min() < 38 or got.max() > 40

    def test_a_constant_q_histogram_has_no_segments_to_take(self):
        from uacpy import acoustic_signal as S
        from uacpy.core.exceptions import ConfigurationError
        r = S.probabilistic_constant_q(self._noise(), self.FS, freq_min=40.0,
                                       bins_per_octave=3)
        assert r.segment_levels_dB is None and r.segment_times is None
        with pytest.raises(ConfigurationError,
                           match='keeps no per-segment levels'):
            r.percentiles(50)


class TestLevelHistogramIsTheArrayForm:

    def test_the_documented_example(self):
        from uacpy.acoustic_signal import level_histogram
        pdf, mean, std = level_histogram(np.array([60.0, 61.0, 61.5, 70.0]),
                                         np.arange(60.0, 64.0, 1.0))
        assert pdf.tolist() == [1 / 3, 2 / 3, 0.0]
        assert mean == 63.125
        assert std == pytest.approx(np.std([60.0, 61.0, 61.5, 70.0]))

    @pytest.mark.parametrize('levels, column_is_nan', [
        ([60.5, 61.5, 70.0], False),     # one level inside: a density
        ([70.0, 80.0, 90.0], True)])     # none inside: no density at all
    def test_only_a_population_with_nothing_inside_is_nan(self, levels,
                                                          column_is_nan):
        from uacpy.acoustic_signal import level_histogram
        pdf, _, _ = level_histogram(np.array(levels), np.arange(60.0, 64.0))
        assert np.isnan(pdf).all() == column_is_nan
        assert np.isnan(pdf).any() == column_is_nan

    def test_the_plot_draws_an_empty_bin_blank(self):
        import matplotlib.pyplot as plt
        from uacpy.acoustic_signal import probabilistic_welch
        r = probabilistic_welch(np.random.default_rng(1).standard_normal(
            48000) * 1e-3, 8000.0, segment_duration=1.0, nperseg=1024)
        fig, ax = r.plot()
        mesh = ax.collections[0].get_array()
        assert np.ma.count_masked(mesh) == int((r.pdf == 0).sum())
        plt.close(fig)

    def test_the_axis_may_be_any_axis(self):
        from uacpy.acoustic_signal import level_histogram
        levels = np.random.default_rng(1).normal(60.0, 3.0, (50, 4))
        edges = np.arange(50.0, 71.0)
        a = level_histogram(levels, edges)
        b = level_histogram(levels.T, edges, axis=1)
        for x, y in zip(a, b):
            assert np.array_equal(x, y, equal_nan=True)
        assert a[0].shape == (20, 4)

    @pytest.mark.parametrize('edges', [[60.0], [61.0, 60.0], [60.0, 60.0],
                                       [[60.0, 61.0]]])
    def test_edges_that_are_not_increasing_bins_are_refused(self, edges):
        from uacpy.acoustic_signal import level_histogram
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='level_edges must be'):
            level_histogram(np.ones(3), edges)

    def test_a_scalar_is_refused(self):
        from uacpy.acoustic_signal import level_histogram
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='got a scalar'):
            level_histogram(60.0, [59.0, 61.0])

    @pytest.mark.parametrize('q, ok', [(0, True), (100, True), ([0, 50], True),
                                       (-0.1, False), (100.1, False),
                                       (np.nan, False)])
    def test_percentiles_must_lie_in_zero_to_one_hundred(self, q, ok):
        from uacpy.acoustic_signal import level_percentiles
        from uacpy.core.exceptions import ConfigurationError
        levels = np.array([[60.0], [62.0], [70.0]])
        if ok:
            assert np.array_equal(level_percentiles(levels, q),
                                  np.percentile(levels, q, axis=0))
        else:
            with pytest.raises(ConfigurationError,
                               match='level_percentiles: q must be'):
                level_percentiles(levels, q)


def test_decidecade_band_levels_takes_frequencies_as_band_levels_does():
    """``band_levels``, ``integrate_psd`` and ``band_level`` take the
    frequencies second, positionally; ``decidecade_band_levels`` refused it
    (keyword-only), so a call copied from its sibling raised TypeError."""
    from uacpy.acoustic_signal import band_levels, decidecade_band_levels
    f = np.linspace(1.0, 5000.0, 4000)
    psd = np.full(f.size, 1e-6)
    by_position = decidecade_band_levels(psd, f)
    by_name = decidecade_band_levels(psd, frequencies=f)
    sibling = band_levels(psd, f)
    np.testing.assert_array_equal(by_position.levels, by_name.levels)
    np.testing.assert_array_equal(by_position.levels, sibling.levels)
