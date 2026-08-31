"""Tests for the carrier ``.plot()`` convenience methods.

Harmonises the plotting API: every uacpy object that renders on its own has a
``.plot()`` that dispatches to its free ``plot_*`` function (mirroring
``Result.plot()``). ``Environment`` / ``SoundSpeedProfile`` plot with no extra
context; ``Absorption`` needs a frequency axis (it *is* a function of
frequency). ``Bottom`` is intentionally excluded — its geoacoustic section is an
environment-level view.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import uacpy
from uacpy.core.absorption import FrancoisGarrison, Thorp


def _env():
    return uacpy.Environment(bathymetry=100.0, ssp=1500.0)


# ── Environment.plot ─────────────────────────────────────────────────────────

def test_environment_plot_returns_fig_ax():
    fig, ax = _env().plot()
    assert isinstance(fig, plt.Figure)


def test_environment_plot_forwards_kwargs():
    fig, ax = _env().plot(title='My env')
    assert ax.get_title() == 'My env'


# ── SoundSpeedProfile.plot ───────────────────────────────────────────────────

def test_ssp_plot_returns_fig_ax():
    ssp = _env().ssp
    fig, ax = ssp.plot()
    assert isinstance(fig, plt.Figure)
    # plot_ssp draws depth increasing downward → y-axis inverted.
    assert ax.yaxis_inverted()


# ── Absorption.plot ──────────────────────────────────────────────────────────

_FREQS = np.logspace(2, 4, 20)          # 100 Hz – 10 kHz


def test_thorp_plot_frequency_curve():
    fig, ax = Thorp().plot(_FREQS)
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert ax.get_ylabel() == 'Absorption (dB/km)'
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), _FREQS)
    assert np.all(line.get_ydata() > 0)


def test_absorption_plot_requires_frequencies():
    with pytest.raises(TypeError):
        Thorp().plot()


def test_francois_garrison_plot_depth_dependent():
    fg = FrancoisGarrison(temperature_c=10, salinity_psu=35, pH=8.0, z_bar_m=0)
    fig, ax = fg.plot(_FREQS, depth=1000.0)
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert np.all(ax.lines[0].get_ydata() > 0)


def test_absorption_plot_forwards_kwargs():
    fig, ax = Thorp().plot(_FREQS, title='α(f)')
    assert ax.get_title(loc='left') == 'α(f)'   # plot_absorption titles left


def test_biological_plot_outside_layer_warns():
    from uacpy.core.absorption import Biological
    bio = Biological(layers=[(40.0, 60.0, 1000.0, 5.0, 10.0)])
    # depth 0 m is outside the 40-60 m layer → α ≡ 0 → blank log-log axes.
    with pytest.warns(UserWarning, match='depth 0'):
        fig, ax = bio.plot(_FREQS, depth=0.0)


def test_biological_plot_inside_layer_no_warning():
    import warnings
    from uacpy.core.absorption import Biological
    bio = Biological(layers=[(40.0, 60.0, 1000.0, 5.0, 10.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fig, ax = bio.plot(_FREQS, depth=50.0)
    assert np.all(ax.lines[0].get_ydata() > 0)


# ── Bathymetry / Altimetry .plot() ───────────────────────────────────────────

def test_bathymetry_plot_points_depth_downward():
    env = uacpy.Environment(bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
                            ssp=1500.0)
    fig, ax = env.bathymetry.plot()
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_xlabel() == 'Range (km)'
    assert ax.yaxis_inverted()


def test_altimetry_plot_keeps_height_upward():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                            altimetry=[(0.0, 0.0), (5000.0, 1.5)])
    fig, ax = env.altimetry.plot()
    # The exact wording comes from the carrier's own ``_VALUE_LABEL``; what
    # matters here is that it is a height in metres and that the axis is NOT
    # inverted (altimetry is positive up, unlike bathymetry).
    assert ax.get_ylabel().lower().endswith('height (m)')
    assert not ax.yaxis_inverted()


def test_range_profile_plot_accepts_title_and_ax():
    env = uacpy.Environment(bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
                            ssp=1500.0)
    fig, ax = plt.subplots()
    _, out = env.bathymetry.plot(ax=ax, title='Seafloor')
    assert out is ax and ax.get_title() == 'Seafloor'


def test_flat_bathymetry_plots_as_a_single_level():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    fig, ax = env.bathymetry.plot()
    assert ax.get_ylabel() == 'Depth (m)'


# ── one spelling for "draw into this Axes" ───────────────────────────────────

class TestEveryPlotMethodSpellsTheAxesArgumentAx:
    """``ax=`` is the name matplotlib uses and the name every other uacpy plot
    method takes. Where it was only reachable through ``**kwargs`` it was
    undocumented and invisible to ``inspect.signature``; where a method spelled
    it ``axes=`` instead, ``ax=`` fell through to ``**kwargs`` and collided
    inside the renderer, raising a ``TypeError`` that named
    ``Result.plot`` — a method the caller never invoked."""

    def _broadband(self):
        from uacpy.core.results import Field, PhaseReference
        freqs = np.linspace(100.0, 500.0, 9)
        return Field(
            data=np.ones((1, 1, freqs.size), dtype=complex),
            coords={'depth': np.array([10.0]), 'range': np.array([1000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([5.0]),
            frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE)

    @pytest.mark.parametrize('owner,method', [
        ('Environment', 'plot'),
        ('SoundSpeedProfile', 'plot'),
        ('Bathymetry', 'plot'),
        ('Altimetry', 'plot'),
        ('Absorption', 'plot'),
        ('Field', 'plot_impulse_response'),
        ('Field', 'plot_transfer_function'),
    ])
    def test_the_parameter_is_named_and_visible(self, owner, method):
        import inspect

        from uacpy.core.absorption import Absorption
        from uacpy.core.altimetry import Altimetry
        from uacpy.core.bathymetry import Bathymetry
        from uacpy.core.environment import Environment
        from uacpy.core.results.field import Field
        from uacpy.core.ssp import SoundSpeedProfile
        owners = {'Environment': Environment, 'SoundSpeedProfile':
                  SoundSpeedProfile, 'Bathymetry': Bathymetry,
                  'Altimetry': Altimetry, 'Absorption': Absorption,
                  'Field': Field}
        parameters = inspect.signature(
            getattr(owners[owner], method)).parameters
        assert 'ax' in parameters, (
            f"{owner}.{method} has no ax parameter; a caller reading the "
            f"signature cannot tell whether it accepts one")

    def test_a_lent_axes_is_drawn_into_by_every_carrier(self):
        env = _env()
        for draw in (lambda ax: env.plot(ax=ax),
                     lambda ax: env.ssp.plot(ax=ax),
                     lambda ax: env.bathymetry.plot(ax=ax),
                     lambda ax: Thorp().plot(np.array([1e3, 1e4]), ax=ax)):
            fig, ax = plt.subplots()
            draw(ax)
            assert ax.lines or ax.collections or ax.patches, draw
            plt.close(fig)

    #: Every spelling of "a pair of Axes" a caller actually has to hand.
    #: ``plt.subplots(2, 1)`` returns an **ndarray**, not a tuple, so a test
    #: that only ever passes ``tuple(pair)`` cannot see a guard that rejects
    #: the thing the docstring tells the reader to build.
    PAIR_SPELLINGS = {
        'subplots-ndarray': lambda axs: axs,
        'ravel-ndarray': lambda axs: axs.ravel(),
        'flat-iterator': lambda axs: axs.flat,
        'tuple': tuple,
        'list': list,
    }

    @pytest.mark.parametrize('spelling', sorted(PAIR_SPELLINGS),
                             ids=sorted(PAIR_SPELLINGS))
    @pytest.mark.parametrize('name', ['ax', 'axes'])
    def test_the_two_panel_plot_takes_a_pair_under_either_name(self, name,
                                                               spelling):
        field = self._broadband()
        fig, pair = plt.subplots(2, 1, sharex=True)
        given = self.PAIR_SPELLINGS[spelling](pair)
        _, drawn = field.plot_transfer_function(**{name: given})
        assert drawn == (pair[0], pair[1])
        assert pair[0].lines and pair[1].lines, 'lent axes were not drawn into'
        plt.close(fig)

    @pytest.mark.parametrize('n_panels', [1, 3])
    def test_anything_but_a_pair_is_a_typed_error(self, n_panels):
        # Both sides of the count boundary, in the spelling matplotlib hands
        # back: two is drawn into by the test above, one and three are named.
        from uacpy.core.exceptions import ConfigurationError
        field = self._broadband()
        fig, axs = plt.subplots(n_panels, 1, squeeze=False)
        with pytest.raises(ConfigurationError,
                           match=f'needs a pair of Axes; got {n_panels}'):
            field.plot_transfer_function(ax=axs.ravel())
        plt.close(fig)

    def test_a_single_axes_for_the_two_panel_plot_is_a_typed_error(self):
        from uacpy.core.exceptions import ConfigurationError
        field = self._broadband()
        fig, ax = plt.subplots()
        with pytest.raises(ConfigurationError, match='needs a pair of Axes'):
            field.plot_transfer_function(ax=ax)
        plt.close(fig)

    def test_the_remediation_names_a_construct_the_guard_accepts(self):
        """The failure this replaced: the message told the reader to build the
        pair from ``plt.subplots(2, 1, sharex=True)``, which the guard then
        refused. Whatever the message names has to work."""
        from uacpy.core.exceptions import ConfigurationError
        field = self._broadband()
        fig, ax = plt.subplots()
        with pytest.raises(ConfigurationError) as excinfo:
            field.plot_transfer_function(ax=ax)
        plt.close(fig)
        assert 'plt.subplots(2, 1, sharex=True)' in str(excinfo.value)

        fig, pair = plt.subplots(2, 1, sharex=True)
        _, drawn = field.plot_transfer_function(ax=pair)
        assert drawn == (pair[0], pair[1])
        plt.close(fig)

    def test_both_spellings_at_once_is_a_typed_error(self):
        from uacpy.core.exceptions import ConfigurationError
        field = self._broadband()
        fig, pair = plt.subplots(2, 1, sharex=True)
        with pytest.raises(ConfigurationError, match='not both'):
            field.plot_transfer_function(ax=tuple(pair), axes=tuple(pair))
        plt.close(fig)


# ── Source.plot_beam_pattern ─────────────────────────────────────────────────
#
# Named rather than spelled ``plot()``: a Source's other rendering is the marker
# ``env.plot(source=...)`` draws, which needs an environment to sit in, so the
# bare name would not say which view was meant.

def _beam_pattern(half_width=15.0, floor=-30.0):
    angles = np.linspace(-180.0, 180.0, 361)
    levels = np.where(np.abs(angles) <= half_width, 0.0, floor)
    return np.column_stack([angles, levels])


def _beamed_source(pattern):
    return uacpy.Source(depths=25.0, frequencies=200.0, beam_pattern=pattern)


def test_source_method_draws_its_own_beam_pattern():
    pattern = _beam_pattern()
    fig, ax = _beamed_source(pattern).plot_beam_pattern()
    assert np.allclose(np.rad2deg(ax.lines[0].get_xdata()), pattern[:, 0])


def test_source_method_forwards_kwargs():
    fig, ax = _beamed_source(_beam_pattern()).plot_beam_pattern(title='Projector')
    assert ax.get_title() == 'Projector'


def test_source_method_on_omni_source_draws_the_flat_circle():
    fig, ax = uacpy.Source(depths=25.0, frequencies=200.0).plot_beam_pattern()
    assert np.allclose(ax.lines[0].get_ydata(), 0.0)


def test_source_method_resolves_a_path_pattern(tmp_path):
    from uacpy.io import write_source_beam_pattern
    pattern = _beam_pattern()
    sbp = tmp_path / 'src.sbp'
    write_source_beam_pattern(sbp, pattern[:, 0], pattern[:, 1])
    fig, ax = _beamed_source(sbp).plot_beam_pattern()
    assert np.allclose(np.rad2deg(ax.lines[0].get_xdata()), pattern[:, 0])


def test_source_method_accepts_an_existing_axis():
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    out_fig, out_ax = _beamed_source(_beam_pattern()).plot_beam_pattern(ax=ax)
    assert out_ax is ax
    assert out_fig is fig
