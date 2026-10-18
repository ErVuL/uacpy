"""Smoke tests for the f-k / Radon / tau-p plotters, plus a documented-
signature + minimal-call sweep of the other ``plots.signal`` free plotters.

Each transform is fed a synthetic array-record and its plotter checked for an
image artist and for honouring ``ax=``. The transforms' own numerics live in
``test_gather_transforms.py``; nothing here asserts on
the values drawn.
"""

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from uacpy.acoustic_signal.gathers import (
    fk_transform, radon_transform, taup_transform,
)
from uacpy.acoustic_signal import BandLevels
from uacpy.core.exceptions import ConfigurationError
from uacpy.plot import plot_beam_power
from uacpy.visualization.plots.signal import (
    plot_fk, plot_radon, plot_taup,
    plot_band_levels, plot_coherence, plot_frf,
    plot_lsfir_diagnostics, plot_psd)


def test_plot_fk_returns_fig_ax():
    f, k, p, _ = fk_transform(
        np.random.default_rng(0).standard_normal((128, 32)), 1000.0, 5.0)
    fig, ax = plot_fk(f, k, p, scaling="density", sound_speed=1500.0, title="t")
    assert fig is not None and ax.images
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    _, ax3 = plot_fk(f, k, p, ax=ax2, scaling="density")
    assert ax3 is ax2
    plt.close(fig2)


def test_plot_radon():
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    mo = np.linspace(-1e-5, 1e-5, 41)
    m, taus, R = radon_transform(d, 1000.0, 10.0, mo, kind="parabolic")
    fig, ax = plot_radon(m, taus, R, kind="parabolic")
    assert ax.images
    plt.close(fig)


def test_plot_radon_needs_the_family_of_bare_arrays():
    from uacpy.core.exceptions import ConfigurationError
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    m, taus, R = radon_transform(d, 1000.0, 10.0, np.linspace(0, 1e-5, 5),
                                 kind="parabolic")
    with pytest.raises(ConfigurationError, match='pass kind='):
        plot_radon(m, taus, R)
    with pytest.raises(ConfigurationError,
                       match='not a Radon moveout family'):
        plot_radon(m, taus, R, kind='linear')
    assert not plt.get_fignums()


class TestRadonResultCarriesItsKind:
    """``RadonResult`` keeps the moveout family it was scanned in, so its
    one-call ``.plot()`` names and scales the axis it holds: a parabolic
    panel drew as "Slowness p (s/km)" at 1e3 instead of curvature at 1e6."""

    @staticmethod
    def _panel(kind='parabolic'):
        d = np.zeros((256, 24))
        d[60, :] = 1.0
        return radon_transform(d, 1000.0, 10.0,
                               np.linspace(0.0, 1e-5, 21), kind=kind)

    def test_the_result_reports_the_kind_it_was_scanned_in(self):
        res = self._panel()
        assert res.kind == 'parabolic'
        moveout, taus, panel = res
        assert moveout.shape == (21,) and panel.shape == (21, 256)

    def test_pickle_and_replace_keep_the_kind(self):
        import pickle
        res = self._panel()
        assert pickle.loads(pickle.dumps(res)).kind == 'parabolic'
        assert res._replace(panel=res.panel * 2).kind == 'parabolic'

    def test_the_one_call_plot_draws_the_curvature_axis(self):
        fig, ax = self._panel().plot()
        try:
            assert 'Curvature' in ax.get_xlabel()
            assert ax.get_xlim()[1] == pytest.approx(10.0, rel=0.05)
        finally:
            plt.close(fig)

    def test_an_unknown_kind_is_refused(self):
        from uacpy.acoustic_signal.gathers import RadonResult
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='kind must be one of'):
            RadonResult(np.zeros(2), np.zeros(3), np.zeros((2, 3)),
                        kind='parabolix')


def test_plot_taup():
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    p, taus, tp = taup_transform(d, 1000.0, 10.0, n_slowness=41, p_max=1.5e-3)
    fig, ax = plot_taup(p, taus, tp, sound_speed=1500.0)
    assert ax.images
    plt.close(fig)


def test_the_slowness_label_stays_inside_the_axes():
    """Example 30's geometry: the 1500 m/s line sits at 0.67 of a 0.83 s/km
    half-axis in a narrow panel, where a label written outward ran past the
    right edge. It is written on the line's p = 0 side."""
    d = np.zeros((512, 48))
    d[100, :] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # the aliasing notice
        p, taus, tp = taup_transform(d, 1000.0, 10.0, n_slowness=301,
                                     p_max=1 / 1200.0)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
    try:
        _, ax = plot_taup(p, taus, tp, ax=axes[0, 1], sound_speed=1500.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = ax.get_window_extent(renderer=renderer)
        (text,) = ax.texts
        drawn = text.get_window_extent(renderer=renderer)
        assert text.get_text().strip() == '1500 m/s'
        assert box.x0 <= drawn.x0 and drawn.x1 <= box.x1
    finally:
        plt.close(fig)


_FREQS3 = np.array([10.0, 100.0, 1000.0])


@pytest.mark.parametrize('fn, params, args', [
    (plot_band_levels, ('centers', 'levels', 'ax', 'lower', 'upper'),
     (BandLevels(np.array([63.0, 80.0, 100.0]), np.array([90.0, 95.0, 92.0]),
                 lower=np.array([56.2, 71.0, 89.1]),
                 upper=np.array([70.8, 89.1, 112.0]),
                 band_type='decidecade', ref=1e-6),)),
    (plot_beam_power, ('angles_deg', 'power', 'ax', 'dB', 'label'),
     (np.linspace(-90.0, 90.0, 7), np.linspace(1.0, 2.0, 7))),
    (plot_frf, ('frequencies', 'tf', 'ax', 'tag'),
     (_FREQS3, np.array([1.0 + 1.0j, 2.0 + 0.0j, 0.5 - 0.5j]))),
    (plot_coherence, ('frequencies', 'coherence', 'ax'),
     (_FREQS3, np.array([0.9, 0.95, 0.99]))),
    (plot_lsfir_diagnostics, ('information_matrix', 'information_vector',
                              'impulse_response', 'fig'),
     (np.eye(3), np.arange(3.0), np.linspace(0.0, 1.0, 4))),
])
def test_signal_plotter_signature_and_smoke(fn, params, args):
    # Each plotter exposes its documented parameters and draws from a
    # minimal call; plot_frf returns a 2-tuple of axes and
    # plot_lsfir_diagnostics a 3-panel list, so axes are flattened.
    sig = inspect.signature(fn)
    assert all(p in sig.parameters for p in params)
    fig, axes = fn(*args)
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        assert ax.has_data()
    plt.close(fig)


def test_one_beam_power_plotter_takes_a_result_or_arrays():
    """``plot_beam_power`` draws beam power in dB re its own maximum
    against look angle from a ``BeamformedField`` or from an angular
    spectrum's two arrays (Bartlett, MVDR, MUSIC), labelled alike; ``dB=False``
    draws it linear, and ``label=`` overlays several processors."""
    angles = np.linspace(-90.0, 90.0, 7)
    fig, ax = plot_beam_power(angles, np.linspace(1.0, 2.0, 7),
                              label='Bartlett')
    plot_beam_power(angles, np.linspace(2.0, 1.0, 7), ax=ax, label='MVDR')
    assert ax.get_ylabel() == 'Beam power (dB re max)'
    assert ax.get_xlabel() == 'Look angle (°)'
    assert np.max(ax.lines[0].get_ydata()) == 0.0
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        'Bartlett', 'MVDR']
    plt.close(fig)
    fig, ax = plot_beam_power(angles, np.linspace(1.0, 2.0, 7), dB=False)
    np.testing.assert_allclose(ax.lines[0].get_ydata(),
                               np.linspace(1.0, 2.0, 7) / 2.0)
    assert ax.get_ylabel() == 'Beam power (re max)'
    plt.close(fig)
    with pytest.raises(ConfigurationError, match='at= picks a point'):
        plot_beam_power(angles, np.ones(7), at=0)


def test_a_null_look_angle_sits_on_the_floor_of_the_beam_plotter():
    """A look angle with zero beam power is ``-inf`` dB; it is drawn on the
    120 dB floor, from a BeamformedField and from arrays alike, the one null
    policy of the dB lines."""
    import warnings
    from uacpy.acoustic_signal import BeamformedField
    response = np.array([1.0, 0.5, 0.0, 0.5, 1.0], dtype=complex)
    beams = BeamformedField(response, np.linspace(-60.0, 60.0, 5),
                            np.ones(1), None)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig, ax = beams.plot()
    try:
        level = ax.lines[0].get_ydata()
        assert np.all(np.isfinite(level))
        assert level[2] == pytest.approx(-120.0)
        assert level[0] == pytest.approx(0.0)
    finally:
        plt.close(fig)
    fig, ax = plot_beam_power(np.linspace(-60.0, 60.0, 5),
                              np.abs(response) ** 2)
    try:
        assert ax.lines[0].get_ydata()[2] == pytest.approx(-120.0)
    finally:
        plt.close(fig)


class TestBeamPowerTakesAPointByItsCoordinate:
    """``grid_coords`` rides on a BeamformedField; ``at={'range': 4000.0}``
    raised an IndexError dressed as 'check the arrays are non-empty', and
    ``at=20`` drew a curve whose title said nothing of which range."""

    @staticmethod
    def _beams():
        from uacpy.acoustic_signal import beamform_field
        positions = np.arange(8) * 0.75
        ranges = np.linspace(1000.0, 5000.0, 5)
        p = np.exp(1j * np.outer(np.arange(8), np.linspace(0.0, 1.0, 5)))
        return beamform_field(p, positions, np.linspace(-90, 90, 37), 1000.0,
                              grid_coords={'range': ranges}), ranges

    @pytest.mark.parametrize('asked, drawn', [(3990.0, 4000.0),
                                              (3400.0, 3000.0)])
    def test_a_coordinate_draws_its_nearest_sample(self, asked, drawn):
        beams, ranges = self._beams()
        fig, ax = beams.plot(at={'range': asked})
        index = int(np.flatnonzero(ranges == drawn)[0])
        _, by_index = beams.plot(at=index)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(),
                                      by_index.lines[0].get_ydata())
        assert ax.get_title() == f'Beam power at range={drawn:g}'
        plt.close('all')

    def test_an_axis_the_grid_does_not_have_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        beams, _ = self._beams()
        with pytest.raises(ConfigurationError, match=r"at= names \['depth'\]"):
            beams.plot(at={'depth': 50.0})


def test_plot_lsfir_diagnostics_is_figure_level():
    # It builds its own three-panel figure, or fills a fig= handed to it,
    # and takes no ax=.
    assert 'ax' not in inspect.signature(plot_lsfir_diagnostics).parameters
    parent = plt.figure()
    fig, axes = plot_lsfir_diagnostics(np.eye(3), np.arange(3.0),
                                       np.linspace(0.0, 1.0, 4), fig=parent)
    assert fig is parent and len(axes) == 3
    plt.close(parent)


class TestTheFRFPlottersTakeTheResultWhole:
    """``plot_coherence`` and ``plot_lsfir_diagnostics`` read their arrays
    off the ``FRFResult`` that holds them, under the result's own names."""

    @staticmethod
    def _pair():
        rng = np.random.default_rng(3)
        x = rng.standard_normal(4096)
        return x, np.convolve(x, [1.0, 0.5, -0.25])[:x.size]

    def test_coherence_from_a_welch_result(self):
        from uacpy.acoustic_signal import etfe, frf_welch
        x, y = self._pair()
        r = frf_welch(x, y, 1000.0, nperseg=256)
        fig, ax = plot_coherence(r)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), r.coherence)
        plt.close(fig)
        with pytest.raises(ConfigurationError, match='carries no coherence'):
            plot_coherence(etfe(x, y, 1000.0))

    def test_diagnostics_from_an_lsfir_result(self):
        from uacpy.acoustic_signal import lsfir
        x, y = self._pair()
        r = lsfir(x, y, 1000.0, order=4)
        fig, axes = plot_lsfir_diagnostics(r)
        np.testing.assert_array_equal(axes[2].lines[0].get_ydata(),
                                      r.impulse_response)
        plt.close(fig)


class TestFixedYWindowsAreEscapable:
    """``plot_coherence`` pins the ordinate to the band its quantity
    normally occupies, and ``plot_psd`` takes a pinned window from
    ``ymin`` / ``ymax`` (its default is fitted to the levels drawn). A pinned
    window renders a record outside it as an empty panel, so the window has
    to be reachable, and a record that falls entirely outside it has to say
    so rather than look like "no data"."""

    _F = np.array([10.0, 100.0, 1000.0])

    def test_plot_coherence_takes_a_level_window(self):
        fig, ax = plot_coherence(self._F, np.full(3, 0.2), ymin=0.0)
        assert ax.get_ylim() == pytest.approx((0.0, 1.01))
        plt.close(fig)

    def test_every_level_axis_is_spelled_ymin_ymax(self):
        """One spelling for the level-axis window across the line plotters
        (``ylim=`` on two of them was the odd one out), and ``freq_min``/``freq_max``
        for the spectrogram's frequency axis, whose level window is
        ``vmin``/``vmax`` like every image plotter's."""
        from uacpy.visualization import plots
        for name in ('plot_psd', 'plot_constant_q_psd', 'plot_ppsd',
                     'plot_constant_q_ppsd', 'plot_band_levels', 'plot_frf',
                     'plot_coherence', 'plot_wenz'):
            params = inspect.signature(getattr(plots, name)).parameters
            assert {'ymin', 'ymax'} <= set(params), name
            assert 'ylim' not in params, name
        params = inspect.signature(plots.plot_spectrogram).parameters
        assert {'freq_min', 'freq_max', 'vmin', 'vmax'} <= set(params)
        assert not {'ymin', 'ymax'} & set(params)

    def test_low_coherence_outside_the_default_window_warns(self):
        with pytest.warns(UserWarning, match=r"outside the plotted y range"):
            fig, ax = plot_coherence(self._F, np.full(3, 0.2))
        plt.close(fig)

    def test_a_quiet_psd_outside_a_pinned_window_warns(self):
        # 1e-14 Pa²/Hz is -20 dB re 1 µPa²/Hz — entirely below a 0-150 dB
        # window.
        with pytest.warns(UserWarning, match=r"Pass ymin=/ymax="):
            fig, ax = plot_psd(self._F, np.full(3, 1e-14), ymin=0.0,
                               ymax=150.0)
        plt.close(fig)

    def test_an_ordinary_record_warns_about_nothing(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = plot_psd(self._F, np.full(3, 1.0))
            plt.close(fig)
            fig, ax = plot_coherence(self._F, np.full(3, 0.98))
            plt.close(fig)


def test_an_angular_spectrum_null_is_drawn_on_its_floor_without_a_warning():
    """A zero-power look direction is drawn at the floor below the peak,
    not dropped with numpy's divide-by-zero warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig, ax = plot_beam_power(np.linspace(-90.0, 90.0, 7),
                                  [0, 1, 2, 4, 2, 1, 0])
    drawn = ax.lines[0].get_ydata()
    assert drawn[0] == drawn[-1] == pytest.approx(-120.0)
    plt.close(fig)


def test_a_transfer_function_null_is_drawn_on_its_floor():
    fig, (ax_mag, _) = plot_frf(np.array([10.0, 100.0, 1000.0]),
                                np.array([1.0 + 0j, 0.0, 0.5]),
                                dynamic_range_dB=80.0)
    assert ax_mag.lines[0].get_ydata()[1] == pytest.approx(-80.0)
    plt.close(fig)


def test_an_undefined_transfer_function_bin_stays_blank_and_sets_no_floor():
    """A NaN bin (no input energy there) is drawn blank; the floor of the
    other bins still follows their own peak."""
    fig, (ax_mag, _) = plot_frf(np.array([10.0, 100.0, 1000.0, 2000.0]),
                                np.array([1.0 + 0j, np.nan, 0.0, 0.1]),
                                dynamic_range_dB=60.0)
    drawn = ax_mag.lines[0].get_ydata()
    assert np.isnan(drawn[1])
    np.testing.assert_allclose(drawn[[0, 2, 3]], [0.0, -60.0, -20.0],
                               atol=1e-12)
    plt.close(fig)


def test_the_frf_phase_is_in_radians_and_the_magnitude_autoscales():
    """One convention for a transfer function: phase in radians wrapped to
    (-π, π], log frequency, |H| autoscaled to what is drawn (a gain of
    +80 dB is in view without a window of the caller's)."""
    f = np.array([10.0, 100.0, 1000.0])
    tf = np.array([1e4 * np.exp(0.5j), 1.0 + 0j, 1e-2 * np.exp(-3.0j)])
    fig, (ax_mag, ax_phase) = plot_frf(f, tf)
    np.testing.assert_allclose(ax_phase.lines[0].get_ydata(), np.angle(tf))
    assert ax_phase.get_ylabel() == 'Phase (rad)'
    assert ax_phase.get_ylim() == pytest.approx((-np.pi, np.pi))
    assert ax_mag.get_xscale() == ax_phase.get_xscale() == 'log'
    lo, hi = ax_mag.get_ylim()
    assert lo <= -40.0 and hi >= 80.0
    plt.close(fig)
    fig, (ax_mag, _) = plot_frf(f, tf, ymin=-10.0, ymax=10.0)
    assert ax_mag.get_ylim() == (-10.0, 10.0)
    plt.close(fig)
