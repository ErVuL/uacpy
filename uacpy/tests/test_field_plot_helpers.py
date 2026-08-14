"""Tests for the Field transfer-function / impulse-response plot helpers and
the ``plot_environment(title=)`` keyword.

Synthetic broadband Fields (no model runs) exercise the reduce-then-plot
convenience methods: a single-receiver field plots directly (singleton axes
auto-squeezed), a multi-receiver field must be reduced with ``.at()`` first,
and a non-broadband field is rejected.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, Modes, PhaseReference
from uacpy.models.sources import model_source

_FREQS = np.linspace(100.0, 300.0, 32)


def _broadband(n_depth=1, n_range=1):
    """Synthetic broadband ``H(depth, range, frequency)`` Field."""
    depths = np.linspace(10.0, 40.0, n_depth)
    ranges = np.linspace(500.0, 2000.0, n_range)
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_depth, n_range, _FREQS.size))
            + 1j * rng.standard_normal((n_depth, n_range, _FREQS.size)))
    return Field(
        data=data,
        coords={'depth': depths, 'range': ranges, 'frequency': _FREQS},
        model='Synthetic', source_depths=np.array([5.0]),
        frequencies=_FREQS, phase_reference=PhaseReference.TRAVELLING_WAVE,
        model_source=model_source('acoustics_toolbox'),
        metadata={'c0': 1500.0},
    )


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close('all')


# ── plot_transfer_function ───────────────────────────────────────────────────

def test_tf_stacked_mag_db_and_phase():
    # Two stacked panels: modulus in dB (top) over phase (bottom), sharing
    # the frequency axis. A single-receiver field plots with no .at().
    fig, (ax_mag, ax_phase) = _broadband().plot_transfer_function()
    assert ax_mag.get_ylabel() == '|H| (dB)'
    assert ax_phase.get_ylabel() == 'Phase (rad)'
    assert ax_phase.get_xlabel() == 'Frequency (Hz)'
    assert len(ax_mag.lines) == 1 and len(ax_phase.lines) == 1
    assert ax_mag.lines[0].get_xdata().shape == _FREQS.shape


def test_tf_after_at_on_grid():
    H = _broadband(n_depth=3, n_range=4)
    fig, (ax_mag, ax_phase) = (
        H.at(depth=25.0, range=1000.0).plot_transfer_function())
    assert ax_phase.get_xlabel() == 'Frequency (Hz)'


def test_tf_title_on_top_panel():
    fig, (ax_mag, ax_phase) = _broadband().plot_transfer_function(title='H(f)')
    assert ax_mag.get_title() == 'H(f)'
    assert ax_phase.get_title() == ''         # title/subtitle only on top


def test_tf_carries_model_credit():
    # The model-source footnote every result plot shows must appear on the TF.
    fig, _ = _broadband().plot_transfer_function()
    assert any('Model' in t.get_text() for t in fig.texts)


def test_tf_multireceiver_without_reduction_raises():
    with pytest.raises(ConfigurationError, match='reduce to one'):
        _broadband(n_depth=3, n_range=4).plot_transfer_function()


def test_tf_non_broadband_raises():
    ts = Field(
        data=np.zeros((1, 1, 8)),
        coords={'depth': [10.0], 'range': [500.0], 'time': np.arange(8) * 0.1},
        model='Synthetic', source_depths=np.array([5.0]),
        frequencies=np.array([200.0]),
        phase_reference=PhaseReference.TRAVELLING_WAVE)
    with pytest.raises(ConfigurationError, match="'frequency'"):
        ts.plot_transfer_function()


# ── plot_impulse_response ────────────────────────────────────────────────────

def test_ir_single_receiver_no_at_needed():
    fig, ax = _broadband().plot_impulse_response()
    assert ax.get_xlabel() == 'Time (s)'
    assert len(ax.lines) == 1


def test_ir_after_at_on_grid():
    H = _broadband(n_depth=3, n_range=4)
    fig, ax = H.at(depth=25.0, range=1000.0).plot_impulse_response()
    assert ax.get_xlabel() == 'Time (s)'


def test_ir_title_passthrough():
    fig, ax = _broadband().plot_impulse_response(title='p(t)')
    assert ax.get_title() == 'p(t)'


def test_ir_carries_model_credit():
    fig, ax = _broadband().plot_impulse_response()
    assert any('Model' in t.get_text() for t in fig.texts)


def test_ir_multireceiver_without_reduction_raises():
    with pytest.raises(ConfigurationError, match='reduce to one'):
        _broadband(n_depth=3, n_range=4).plot_impulse_response()


def test_ir_unpinned_range_raises():
    # A frequency-only Field with no pinned range cannot be IFFTed honestly
    # (t_start / demodulation need r); fabricating r=0 mispositions the peak.
    spec = Field(
        data=np.exp(-2j * np.pi * _FREQS * 2.0),
        coords={'frequency': _FREQS},
        model='Synthetic', source_depths=np.array([5.0]),
        frequencies=_FREQS, phase_reference=PhaseReference.TRAVELLING_WAVE)
    with pytest.raises(ConfigurationError, match='pinned range|range'):
        spec.plot_impulse_response()
    assert not plt.get_fignums()                        # no leaked figure


def test_tf_real_field_raises_before_drawing():
    # A real (already-dB) broadband field has no phase panel; the guard must
    # be a typed error raised before any figure exists.
    tl = _broadband().to_db()
    with pytest.raises(ConfigurationError, match='complex'):
        tl.plot_transfer_function()
    assert not plt.get_fignums()


# ── compare guards + credits ─────────────────────────────────────────────────

def test_compare_label_length_mismatch_raises():
    a = _broadband(n_depth=2, n_range=4).at(depth=25.0, range=1000.0)
    with pytest.raises(ConfigurationError, match='labels'):
        uacpy.plot.compare([a, a, a], labels=['one', 'two'])


def test_compare_draws_model_credit():
    a = _broadband().isel(depth=0, range=0)
    fig, ax = uacpy.plot.compare([a, a], labels=['A', 'B'])
    assert any('Model' in t.get_text() for t in fig.texts)


# ── plot_environment(title=) ─────────────────────────────────────────────────

def test_environment_plot_accepts_title():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    fig, ax = env.plot(title='My environment')
    assert ax.get_title() == 'My environment'


# ── env= governs the plotted depth extent ────────────────────────────────────

def _tl_field(max_depth=20.0):
    """A (depth, range) TL field whose receiver grid stops above the seabed."""
    z = np.linspace(1.0, max_depth, 12)
    r = np.linspace(1.0, 2000.0, 20)
    rng = np.random.default_rng(0)
    return Field(
        data=60.0 + 10.0 * rng.random((z.size, r.size)),
        coords={'depth': z, 'range': r},
        model='Synthetic', frequencies=np.array([1000.0]),
    )


def test_env_extends_depth_axis_to_the_seabed():
    """A receiver grid ending above the seafloor still plots the full water
    column once the environment is supplied."""
    env = uacpy.Environment(bathymetry=30.0, ssp=1500.0)
    fig, ax = _tl_field().plot(env=env)
    assert max(ax.get_ylim()) > 30.0


def test_without_env_the_plot_spans_only_the_data():
    """A result carries no environment, so on its own it can only span the
    grid it was sampled on."""
    fig, ax = _tl_field().plot()
    assert max(ax.get_ylim()) < 21.0


def test_env_on_a_view_that_cannot_use_it_raises():
    """Silently swallowing env= would look like it had an effect."""
    modes = Modes(
        k=np.array([0.4, 0.3, 0.2]) + 0j,
        phi=np.random.default_rng(0).random((8, 3)),
        depths=np.linspace(0.0, 100.0, 8),
        model='Synthetic', frequencies=np.array([100.0]),
    )
    with pytest.raises(ConfigurationError, match='env='):
        modes.plot(env=uacpy.Environment(bathymetry=100.0, ssp=1500.0))


# ── depth-axis inversion is idempotent under ax= composition ─────────────────

def test_repeated_plots_into_one_ax_keep_depth_pointing_down():
    """``invert_yaxis`` toggles; ``ax=`` is the documented overlay hook."""
    f = _tl_field()
    fig, ax = plt.subplots()
    f.at(range=1.0).plot(ax=ax)
    f.at(range=2000.0).plot(ax=ax)
    assert ax.yaxis_inverted()


# ── source= / receiver= geometry on a TL heatmap ─────────────────────────────

def _geometry():
    src = uacpy.Source(depths=20.0, frequencies=1000.0)
    rcv = uacpy.Receiver(depths=np.linspace(1.0, 20.0, 12),
                         ranges=np.linspace(1.0, 2000.0, 20))
    return src, rcv


def test_plot_field_draws_source_and_receiver():
    """The same geometry ``env.plot`` and the ray plotter draw is available on
    a TL heatmap."""
    src, rcv = _geometry()
    env = uacpy.Environment(bathymetry=30.0, ssp=1500.0)
    bare = _tl_field().plot(env=env)[1]
    with_geom = _tl_field().plot(env=env, source=src, receiver=rcv)[1]
    assert len(with_geom.lines) > len(bare.lines)


def test_geometry_markers_land_at_the_source_depth():
    src, rcv = _geometry()
    fig, ax = _tl_field().plot(source=src)
    ys = [float(ln.get_ydata()[0]) for ln in ax.lines if len(ln.get_ydata())]
    assert any(abs(y - 20.0) < 1e-9 for y in ys)


@pytest.mark.parametrize('kwargs', [{'source': True}, {'receiver': True}])
def test_geometry_on_a_1d_cut_is_rejected_not_ignored(kwargs):
    src, rcv = _geometry()
    real = {'source': src, 'receiver': rcv}
    key = next(iter(kwargs))
    with pytest.raises(ConfigurationError, match=f'{key}='):
        _tl_field().at(depth=10.0).plot(**{key: real[key]})
