"""The Field transfer-function / impulse-response plot methods, and the
``plot_environment(title=)`` keyword.

Synthetic broadband Fields (no model runs) exercise the reduce-then-plot
convenience methods: a single-receiver field plots directly (singleton axes
auto-squeezed), a multi-receiver field must be reduced with ``.at()`` first,
and a non-broadband field is rejected.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, Modes, PhaseReference, SoundSpeeds
from uacpy.models.provenance import model_provenance

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
        model_source=model_provenance('acoustics_toolbox'),
        speeds=SoundSpeeds(surface=1500.0),
    )


# ── plot_transfer_function ───────────────────────────────────────────────────

def test_tf_stacked_mag_dB_and_phase():
    # Two stacked panels: modulus in dB (top) over phase (bottom), sharing
    # the frequency axis. A single-receiver field plots with no .at().
    fig, (ax_mag, ax_phase) = _broadband().plot_transfer_function()
    assert ax_mag.get_ylabel() == '|H| (dB)'
    assert ax_phase.get_ylabel() == 'Phase (rad)'
    assert ax_phase.get_xlabel() == 'Frequency (Hz)'
    assert len(ax_mag.lines) == 1 and len(ax_phase.lines) == 1
    assert ax_mag.lines[0].get_xdata().shape == _FREQS.shape


def test_the_spectrum_of_a_received_trace_is_captioned_s_times_h():
    """The transform of a received pressure trace is S·H (Pa·s), the source
    spectrum times the channel: its dB view is captioned -20·log10|S·H|, not
    TL, and its level view |S·H|; a transfer function H keeps its captions."""
    f = np.linspace(100.0, 1000.0, 10)
    data = 1e-3 * np.exp(1j * f / 100.0)
    spectrum = Field(data=data, coords={'frequency': f}, kind='pressure',
                     unit='Pa·s')
    fig, ax = spectrum.plot()
    assert ax.get_ylabel() == '−20·log10|S·H| (dB re 1 Pa·s)'
    plt.close(fig)
    fig, (ax_mag, _) = spectrum.plot_transfer_function()
    assert ax_mag.get_ylabel() == '|S·H| (dB re 1 Pa·s)'
    plt.close(fig)
    transfer = Field(data=data, coords={'frequency': f})
    fig, ax = transfer.plot()
    assert ax.get_ylabel() == 'TL (dB)'
    plt.close(fig)
    fig, (ax_mag, _) = transfer.plot_transfer_function()
    assert ax_mag.get_ylabel() == '|H| (dB)'
    plt.close(fig)


def test_the_two_frf_plotters_label_and_frame_the_pair_alike():
    """Whichever plotter draws an H(f) pair, the magnitude panel carries the
    same label, the phase panel the whole (-π, π] window, and one legend
    sits on the magnitude panel."""
    from uacpy.plot import plot_frf
    fig_tf, (mag_tf, phase_tf) = _broadband().plot_transfer_function(
        label='model')
    fig_frf, (mag_frf, phase_frf) = plot_frf(
        _FREQS, np.exp(1j * np.linspace(0.0, 1.0, _FREQS.size)),
        label='measured')
    try:
        assert mag_tf.get_ylabel() == mag_frf.get_ylabel() == '|H| (dB)'
        for phase in (phase_tf, phase_frf):
            assert phase.get_ylim() == pytest.approx((-np.pi, np.pi))
            assert phase.get_legend() is None
        assert mag_tf.get_legend() is not None
        assert mag_frf.get_legend() is not None
    finally:
        plt.close(fig_tf)
        plt.close(fig_frf)


def test_the_two_frf_plotters_floor_size_and_title_the_pair_alike():
    """One H(f) through either plotter: a null deeper than 120 dB below the
    peak on the same floor (``dynamic_range_dB`` on both), the same figure
    size, and the same default caption, the Field's naming its cell."""
    from uacpy.plot import plot_frf
    tf = np.exp(1j * np.linspace(0.0, 1.0, _FREQS.size))
    tf[10] = 1e-9                                     # a 180 dB-deep null
    H = Field(data=tf[None, None, :],
              coords={'depth': np.array([20.0]), 'range': np.array([1000.0]),
                      'frequency': _FREQS},
              model='Synthetic', speeds=SoundSpeeds(surface=1500.0))
    fig_tf, (mag_tf, _) = H.plot_transfer_function()
    fig_frf, (mag_frf, _) = plot_frf(_FREQS, tf)
    fig_40, (mag_40, _) = H.plot_transfer_function(dynamic_range_dB=40.0)
    try:
        drawn_tf = mag_tf.get_lines()[0].get_ydata()
        np.testing.assert_allclose(drawn_tf,
                                   mag_frf.get_lines()[0].get_ydata())
        assert np.min(drawn_tf) == pytest.approx(-120.0)
        assert np.min(mag_40.get_lines()[0].get_ydata()) == pytest.approx(
            -40.0)
        assert tuple(fig_tf.get_size_inches()) == tuple(
            fig_frf.get_size_inches()) == (8.0, 6.0)
        assert mag_frf.get_title() == 'Frequency response'
        assert mag_tf.get_title().startswith('Frequency response — ')
    finally:
        for fig in (fig_tf, fig_frf, fig_40):
            plt.close(fig)


def test_tf_shares_plot_frfs_axes_convention():
    """Both FRF plotters draw one convention: log frequency, phase in
    radians, |H| autoscaled; ``label=`` names a line and adds a legend, so
    a model overlays a measurement on one pair of axes."""
    from uacpy.plot import plot_frf
    H = _broadband()
    fig, (ax_mag, ax_phase) = H.plot_transfer_function(label='model')
    # Read before the overlay: plot_frf sets both on the axes it is handed.
    assert ax_mag.get_xscale() == ax_phase.get_xscale() == 'log'
    assert ax_phase.get_ylabel() == 'Phase (rad)'
    plot_frf(_FREQS, np.ones(_FREQS.size, dtype=complex), ax=(ax_mag, ax_phase),
             label='measured')
    assert ax_mag.get_xscale() == ax_phase.get_xscale() == 'log'
    assert ax_phase.get_ylabel() == 'Phase (rad)'
    assert [line.get_label() for line in ax_mag.lines] == ['model',
                                                           'measured']
    assert ax_mag.get_legend() is not None
    plt.close(fig)


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

_FS = 1000.0            # Hz: the impulse responses below, a band to 300 Hz


def _ir(**kwargs):
    """The impulse response of the single-receiver synthetic field."""
    return _broadband().to_time_trace(sample_rate=_FS, **kwargs)


def test_ir_draws_the_trace_it_is_handed():
    trace = _ir()
    fig, ax = trace.plot_impulse_response()
    assert ax.get_xlabel() == 'Time (s)'
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), trace.times)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),
                                  np.asarray(trace.data))


def test_ir_draws_one_cell_of_a_grids_impulse_response():
    H = _broadband(n_depth=3, n_range=4)
    cell = H.impulse_response(_FS).at(depth=25.0, range=1000.0)
    fig, ax = cell.plot_impulse_response()
    # The grid's records run on its reduced time, and the trace says so.
    assert ax.get_xlabel() == 'Reduced time t − r/1500 m/s (s)'
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),
                                  np.asarray(cell.data))


def test_ir_refuses_a_transfer_function_naming_the_calls_that_make_h():
    """The plotter draws h(t); it computes no synthesis from an H(f)."""
    with pytest.raises(ConfigurationError,
                       match=r"(?s)h\(t\).*to_time_trace.*"
                             r"impulse_response\(fs\)"):
        _broadband().plot_impulse_response()
    assert not plt.get_fignums()                        # no leaked figure


def test_the_ir_axis_names_an_impulse_response_not_a_pressure():
    """With no source the trace is H integrated over the band, per second,
    not a pressure: its plot reads h(t) in 1/s."""
    _, ax = _ir().plot_impulse_response()
    assert ax.get_ylabel() == 'h(t) (1/s)'
    _, ax2 = _ir().plot()
    assert ax2.get_ylabel() == 'h(t) (1/s)'


def test_a_trace_is_per_second_without_a_source_and_pa_with_one():
    """Both routes of to_time_trace: the bare impulse response is
    kind='impulse_response' in 1/s, and the same call with a source
    waveform is the received pressure in Pa."""
    from uacpy.acoustic_signal import tone_burst
    H = _broadband()
    fs = 4.0 * float(_FREQS[-1])
    bare = H.to_time_trace(sample_rate=fs)
    assert (bare.kind, bare.unit) == ('impulse_response', '1/s')
    _, s = tone_burst(float(_FREQS[_FREQS.size // 2]), 5, sample_rate=fs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        driven = H.to_time_trace(source_waveform=s, sample_rate=fs)
    assert (driven.kind, driven.unit) == ('pressure', 'Pa')
    _, ax = driven.plot()
    assert ax.get_ylabel() == 'p(t) (Pa)'
    # The way back is a transfer function again, not an impulse response.
    assert bare.to_transfer_function().kind == 'pressure'


def _two_path(df):
    """``H = e^{-2πif·0.1} + 0.6·e^{-2πif·0.13}`` over 100-400 Hz, one cell
    at 150 m, on a grid of spacing ``df`` whose bins are multiples of it."""
    freqs = np.arange(100.0, 400.0 + df / 2, df)
    data = (np.exp(-2j * np.pi * freqs * 0.10)
            + 0.6 * np.exp(-2j * np.pi * freqs * 0.13))
    return freqs, data, Field(
        data=data.reshape(1, 1, -1),
        coords={'depth': np.array([10.0]), 'range': np.array([150.0]),
                'frequency': freqs},
        frequencies=freqs, phase_reference=PhaseReference.TRAVELLING_WAVE)


@pytest.mark.parametrize('df', [1.0, 0.5])
def test_the_tone_of_an_impulse_response_is_refused_and_its_transform_is_h(df):
    """A band-limited impulse response is a transient: the tone estimator
    returns 2·Δf·H(f) on it (2.0 × H at Δf = 1 Hz, 1.0 × at 0.5 Hz under
    window=None, a record-dependent fraction under 'hann'), so
    extract_tone refuses it, and the refusal's remedy returns H exactly."""
    freqs, data, H = _two_path(df)
    bare = H.to_time_trace(sample_rate=3200.0, t_start=0.0)
    assert bare.kind == 'impulse_response'
    with pytest.raises(ConfigurationError,
                       match='(?s)impulse response.*to_transfer_function'):
        bare.extract_tone(200.0)
    back = bare.to_transfer_function().at(frequency=200.0)
    expected = data[np.argmin(np.abs(freqs - 200.0))]
    assert abs(complex(back.data) / expected - 1.0) < 1e-12
    assert (back.kind, back.unit) == ('pressure', 'Pa')


def test_a_line_of_impulse_responses_transforms_to_a_line_of_h_in_db():
    """A line of impulse responses (one per range, one shared time base)
    is taken back to H by to_transfer_function, and that line draws in dB
    like any H."""
    H = _broadband(n_range=3)
    f0 = float(_FREQS[_FREQS.size // 2])
    ranges = H.coords['range']
    traces = [H.to_time_trace(depth=10.0, range=float(r), t_start=0.0,
                              sample_rate=_FS) for r in ranges]
    line = Field(data=np.stack([t.data for t in traces]),
                 coords={'range': ranges, 'time': traces[0].coords['time']},
                 kind=traces[0].kind, unit=traces[0].unit,
                 metadata=dict(traces[0].metadata))
    assert line.kind == 'impulse_response'
    tones = line.to_transfer_function().at(frequency=f0)
    assert tones.kind == 'pressure'
    fig, ax = tones.plot(value='dB')
    assert ax.get_ylabel() == 'TL (dB)'
    plt.close(fig)


def test_ir_title_passthrough():
    fig, ax = _ir().plot_impulse_response(title='p(t)')
    assert ax.get_title() == 'p(t)'


def test_ir_carries_model_credit():
    fig, ax = _ir().plot_impulse_response()
    assert any('Model' in t.get_text() for t in fig.texts)


def test_ir_multireceiver_without_reduction_raises():
    grid = _broadband(n_depth=3, n_range=4).impulse_response(_FS)
    with pytest.raises(ConfigurationError, match='Select one cell first'):
        grid.plot_impulse_response()
    assert not plt.get_fignums()                        # no leaked figure


def test_tf_real_field_raises_before_drawing():
    # A real (already-dB) broadband field has no phase panel; the guard must
    # be a typed error raised before any figure exists.
    tl = _broadband().to_dB()
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
    column once the environment is supplied — the axis bottom sits 5 %
    below the seafloor (``env.depth * 1.05``)."""
    env = uacpy.Environment(bathymetry=30.0, ssp=1500.0)
    fig, ax = _tl_field().plot(env=env)
    assert max(ax.get_ylim()) == pytest.approx(30.0 * 1.05)


def test_range_dependent_seafloor_headroom_below_deepest_clipped_point():
    """The bathymetry is clipped to the painted range span — the mesh's cell
    edges, half a cell past the outermost sample — and the depth axis bottom
    sits 5 % below the deepest seafloor point of that span."""
    env = uacpy.Environment(bathymetry=[(0.0, 100.0), (5000.0, 120.0)],
                            ssp=1500.0)
    field = _tl_field()
    fig, ax = field.plot(env=env)
    # The deepest painted seafloor sits at the far cell edge of the field's
    # range grid, on the 100→120 m slope.
    r = field.coords['range']
    far_edge = r[-1] + 0.5 * (r[-1] - r[-2])
    deepest = np.interp(far_edge, [0.0, 5000.0], [100.0, 120.0])
    assert max(ax.get_ylim()) == pytest.approx(deepest * 1.05)


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


def test_the_field_plot_methods_draw_what_the_public_plotters_draw():
    from uacpy.plot import (plot_impulse_response,
                                     plot_transfer_function)
    H = _broadband()
    _, (mag_m, phase_m) = H.plot_transfer_function()
    _, (mag_f, phase_f) = plot_transfer_function(H)
    for by_method, by_plotter in ((mag_m, mag_f), (phase_m, phase_f)):
        assert ([ln.get_xydata().tolist() for ln in by_method.lines]
                == [ln.get_xydata().tolist() for ln in by_plotter.lines])
    _, ax_m = _ir().plot_impulse_response()
    _, ax_f = plot_impulse_response(_ir())
    assert ([ln.get_xydata().tolist() for ln in ax_m.lines]
            == [ln.get_xydata().tolist() for ln in ax_f.lines])
    plt.close('all')
