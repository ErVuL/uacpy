"""Smoke tests for the f-k / Radon / tau-p plotters, plus a documented-
signature + minimal-call sweep of the other ``plots.signal`` free plotters.

Each transform is fed a synthetic array-record and its plotter checked for an
image artist and for honouring ``ax=``. The transforms' own numerics live in
``test_gather_transforms.py``; nothing here asserts on
the values drawn.
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from uacpy.acoustic_signal.gathers import (
    fk_transform, radon_transform, taup_transform,
)
from uacpy.visualization.plots.signal import (
    plot_fk, plot_radon, plot_taup,
    plot_angular_spectrum, plot_band_levels, plot_coherence, plot_frf,
    plot_lsfir_diagnostics, plot_psd)


def test_plot_fk_returns_fig_ax():
    f, k, p, _ = fk_transform(
        np.random.default_rng(0).standard_normal((128, 32)), 1000.0, 5.0)
    fig, ax = plot_fk(f, k, p, scaling="power", sound_speed=1500.0, title="t")
    assert fig is not None and ax.images
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    _, ax3 = plot_fk(f, k, p, ax=ax2, scaling="power")
    assert ax3 is ax2
    plt.close(fig2)


def test_plot_radon():
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    mo = np.linspace(-1.5e-3, 1.5e-3, 41)
    m, taus, R = radon_transform(d, 1000.0, 10.0, mo, kind="linear")
    fig, ax = plot_radon(m, taus, R, kind="linear")
    assert ax.images
    plt.close(fig)


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


_FREQS3 = np.array([10.0, 100.0, 1000.0])


@pytest.mark.parametrize('fn, params, args', [
    (plot_band_levels, ('centers', 'levels', 'ax'),
     (np.array([63.0, 80.0, 100.0]), np.array([90.0, 95.0, 92.0]))),
    (plot_angular_spectrum, ('angles_deg', 'spectrum', 'ax', 'dB'),
     (np.linspace(-90.0, 90.0, 7), np.linspace(1.0, 2.0, 7))),
    (plot_frf, ('frequencies', 'tf', 'ax', 'tag'),
     (_FREQS3, np.array([1.0 + 1.0j, 2.0 + 0.0j, 0.5 - 0.5j]))),
    (plot_coherence, ('frequencies', 'coh', 'ax'),
     (_FREQS3, np.array([0.9, 0.95, 0.99]))),
    (plot_lsfir_diagnostics, ('Minfo', 'Vinfo', 'g'),
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


def test_both_beam_power_plotters_label_the_peak_relative_level_alike():
    """``plot_angular_spectrum`` and ``plot_beam_power`` draw one quantity,
    beam power in dB re its own maximum, against look angle; the axis says
    're max' on both, since a bare 'Power (dB)' hides the reference."""
    fig, ax = plot_angular_spectrum(np.linspace(-90.0, 90.0, 7),
                                    np.linspace(1.0, 2.0, 7))
    assert ax.get_ylabel() == 'Power (dB re max)'
    assert ax.get_xlabel() == 'Look angle (°)'
    assert np.max(ax.lines[0].get_ydata()) == 0.0
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
    # It builds its own three-panel figure and takes no ax=.
    assert 'ax' not in inspect.signature(plot_lsfir_diagnostics).parameters


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
                     'plot_constant_q_ppsd', 'plot_sel', 'plot_frf',
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
