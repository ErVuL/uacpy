"""Pins for the raw-array comms plotters in ``plots.comms``.

``plot_subcarriers``, ``plot_channel`` and ``plot_eye_diagram`` carry real
axis/indexing contracts (OFDM bin addressing, ms delay axis, 0 Hz-centred
spectrum, symbol-interval windowing) and are asserted on the drawn data; the
thin wrappers (``plot_convergence``, ``plot_sync_metric``,
``plot_doppler_ambiguity``, which takes the ``DopplerEstimate`` whole) get
their one transform pinned plus a shared smoke/``ax=`` sweep. The comms
numerics themselves live in ``test_comms.py``
and ``test_comms_equalizers.py``; the constellation/BER plotters are covered in
``test_carrier_plot_methods.py`` callers and the doc figure gates.
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from uacpy.plot import (
    plot_channel, plot_convergence, plot_doppler_ambiguity, plot_eye_diagram,
    plot_subcarriers, plot_sync_metric,
)

# Asymmetric complex taps: |H(f)| has no conjugate symmetry, so any
# fftshift/index-origin slip moves values, not just labels.
_TAPS = np.array([1.0, 0.5j, -0.25])


def test_plot_subcarriers_addresses_bins_like_ofdm_demodulate():
    # ofdm_demodulate equalizes bin k with np.fft.fft(h, nsc)[k]; the plot
    # must put that same value at x = k (unshifted, DC at index 0).
    from uacpy.comms import subcarrier_response
    nsc = 8
    fig, ax = plot_subcarriers(subcarrier_response(_TAPS, nsc))
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), np.arange(nsc))
    expected = 20 * np.log10(np.abs(np.fft.fft(_TAPS, nsc)) + 1e-12)
    np.testing.assert_allclose(line.get_ydata(), expected)
    plt.close(fig)


def test_plot_channel_draws_the_taps_on_one_axes():
    """The delay view only: the frequency response is a result of its own
    (``channel_response(...).plot()``), not a panel this plotter computes."""
    fig, ax = plot_channel(_TAPS, 1000.0)
    assert ax.figure is fig
    assert ax.containers             # one stem per tap
    assert ax.get_title() == 'Channel impulse response'
    plt.close(fig)
    fig, given = plt.subplots()
    out_fig, out_ax = plot_channel(_TAPS, 1000.0, given, title='delay')
    assert out_fig is fig and out_ax is given
    assert given.get_title() == 'delay'
    plt.close(fig)


def test_plot_subcarriers_refuses_a_channel_and_a_count():
    """The plotter computes no DFT: a (channel, count) call is refused, naming
    ``subcarrier_response``."""
    from uacpy.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match='subcarrier_response'):
        plot_subcarriers(_TAPS, 8)
    assert not plt.get_fignums()


def test_plot_channel_delay_axis_is_milliseconds():
    fig, ax = plot_channel(np.array([1.0, 0.5, 0.25, 0.1]), 1000.0)
    stem_x = ax.containers[0].markerline.get_xdata()
    np.testing.assert_allclose(stem_x, [0.0, 1.0, 2.0, 3.0])
    plt.close(fig)


def test_the_channel_response_is_drawn_centred_on_zero_hz():
    from uacpy.acoustic_signal import channel_response
    fs = 500.0
    fig, ax = channel_response(_TAPS, fs).plot()
    f = ax.lines[0].get_xdata()
    mag_dB = ax.lines[0].get_ydata()
    assert np.all(np.diff(f) > 0)
    assert f[0] == -fs / 2
    # H(0) = sum of the taps; misaligned f/H shifts would break this pairing.
    dc = 20 * np.log10(np.abs(_TAPS.sum()) + 1e-12)
    np.testing.assert_allclose(mag_dB[f == 0.0], [dc])
    assert ax.get_title() == 'Frequency response'
    plt.close(fig)


def test_plot_eye_diagram_overlays_sps_strided_windows_in_symbol_units():
    sps, n_symbols = 4, 2
    span = sps * n_symbols
    x = np.arange(20.0)
    fig, ax = plot_eye_diagram(x, sps, n_symbols=n_symbols)
    # Windows [0:8], [4:12], [8:16], [12:20] all fit in 20 samples.
    assert len(ax.lines) == 4
    for k, line in enumerate(ax.lines):
        np.testing.assert_allclose(line.get_xdata(), np.arange(span) / sps)
        np.testing.assert_allclose(line.get_ydata(), x[k * sps: k * sps + span])
    plt.close(fig)


def test_plot_eye_diagram_draws_no_traces_for_a_signal_shorter_than_one_window():
    fig, ax = plot_eye_diagram(np.arange(7.0), 4, n_symbols=2)
    assert len(ax.lines) == 0
    plt.close(fig)


def test_plot_eye_diagram_draws_a_trace_for_a_signal_exactly_one_window_long():
    fig, ax = plot_eye_diagram(np.arange(8.0), 4, n_symbols=2)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_sync_metric_draws_threshold_line_only_when_given():
    metric = np.array([0.1, 0.2, 0.9, 0.2])
    fig, ax = plot_sync_metric(metric)
    assert len(ax.lines) == 1 and ax.get_legend() is None
    plt.close(fig)
    fig, ax = plot_sync_metric(metric, threshold=0.4)
    assert len(ax.lines) == 2 and ax.get_legend() is not None
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.4, 0.4])
    plt.close(fig)


def _doppler_estimate(best_scale, scales, peak):
    from uacpy.comms import DopplerEstimate
    return DopplerEstimate(best_scale, scales, peak)


def test_plot_doppler_ambiguity_plots_scales_in_thousandths_and_marks_the_peak():
    scales = np.linspace(-2e-3, 2e-3, 5)
    peak = np.array([0.1, 0.2, 0.3, 1.0, 0.4])
    fig, ax = plot_doppler_ambiguity(_doppler_estimate(1e-3, scales, peak))
    np.testing.assert_allclose(ax.lines[0].get_xdata(), scales * 1e3)
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 1.0])
    plt.close(fig)


def test_plot_doppler_ambiguity_marks_the_estimators_best_scale():
    """Example 31's probe. The line marks the estimate's own best_scale;
    the plotter computes nothing."""
    from uacpy import comms
    fs = 12000.0
    t = np.arange(4000) / fs
    probe = np.exp(1j * 2 * np.pi * (1500 * t + 0.5 * 2e5 * t ** 2))
    estimate = comms.estimate_doppler_scale(
        comms.compensate_doppler(probe, -1.5e-3), probe,
        np.linspace(0, 3e-3, 61))
    assert estimate.best_scale == pytest.approx(1.5e-3, abs=1e-12)
    for fig, ax in (plot_doppler_ambiguity(estimate), estimate.plot()):
        np.testing.assert_allclose(ax.lines[1].get_xdata(),
                                   [estimate.best_scale * 1e3] * 2)
        assert ax.get_legend().get_texts()[0].get_text() == (
            f"a = {estimate.best_scale * 1e3:.2f} e-3")
        plt.close(fig)


def test_plot_doppler_ambiguity_refuses_bare_arrays():
    from uacpy.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match='needs the DopplerEstimate'):
        plot_doppler_ambiguity(np.linspace(-1e-3, 1e-3, 5))


def test_plot_convergence_plots_mse_in_decibels_with_a_floor():
    mse = np.array([1.0, 0.1, 0.01, 0.0])
    fig, ax = plot_convergence(mse)
    np.testing.assert_allclose(ax.lines[0].get_ydata(),
                               [0.0, -10.0, -20.0, -120.0])
    plt.close(fig)


@pytest.mark.parametrize("plotter, args", [
    (plot_convergence, (np.array([1.0, 0.5, 0.1]),)),
    (plot_sync_metric, (np.array([0.1, 0.8, 0.2]),)),
    (plot_doppler_ambiguity, (_doppler_estimate(
        0.0, np.linspace(-1e-3, 1e-3, 5),
        np.array([0.2, 0.4, 1.0, 0.5, 0.3])),)),
    (plot_subcarriers, (np.fft.fft(_TAPS, 8),)),
])
def test_comms_plotters_return_fig_ax_and_honour_ax(plotter, args):
    fig, ax = plotter(*args)
    assert fig is not None and ax.figure is fig
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    _, ax3 = plotter(*args, ax=ax2)
    assert ax3 is ax2
    plt.close(fig2)


def test_the_quadrature_label_of_a_wide_constellation_stays_on_the_figure():
    """The "Quadrature" axis label sits entirely inside the saved figure even
    with four-digit tick labels on the default 5-inch square panel: the
    plotter lays its own figure out around its labels."""
    from uacpy import comms
    from uacpy.visualization.plots.comms import plot_constellation
    fig, ax = plot_constellation(1000.0 * comms.constellation('16qam'),
                                 scheme='16qam')
    fig.canvas.draw()
    bbox = ax.yaxis.label.get_window_extent()
    assert bbox.x0 >= 0.0, f"ylabel starts {bbox.x0:.1f} px left of the figure"
    plt.close(fig)


def test_an_error_free_point_is_not_drawn_as_a_measured_ber():
    """Zero errors in N bits measures no BER; drawing it on the line at a
    floor of 1e-12 put a fabricated waterfall decades deep. The measured line
    carries only the non-zero points, and with ``n_bits`` each error-free one
    is a hollow downward caret at 1/n_bits; without it, a warning."""
    from uacpy.plot import plot_ber_curve
    ebn0 = np.array([0.0, 4.0, 8.0, 12.0])
    ber = np.array([1e-1, 1e-2, 0.0, 0.0])

    with pytest.warns(UserWarning, match=r"2 point\(s\) have zero errors"):
        fig, ax = plot_ber_curve(ebn0, ber)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1e-1, 1e-2])
    assert len(ax.lines) == 1
    assert ax.get_ylim()[0] > 1e-4
    plt.close(fig)

    fig, ax = plot_ber_curve(ebn0, ber, n_bits=[1e3, 1e3, 1e4, 1e5])
    measured, bound = ax.lines
    np.testing.assert_array_equal(measured.get_xdata(), [0.0, 4.0])
    np.testing.assert_array_equal(bound.get_xdata(), [8.0, 12.0])
    np.testing.assert_allclose(bound.get_ydata(), [1e-4, 1e-5])
    assert bound.get_marker() == 'v' and bound.get_linestyle() == 'None'
    plt.close(fig)


def test_the_theory_curve_does_not_stretch_the_axis_past_the_run():
    """QPSK theory at 20 dB Eb/N0 is ~1e-45: drawn over the whole Eb/N0 span
    it pulled the log axis down ~47 decades, the measurement squeezed into
    the top few. The axis stops one decade below the lowest rate the run
    resolved — its lowest measured BER, or 1/n_bits when lower."""
    from uacpy.comms import BerCurve, ber_theory_curve
    from uacpy.plot import plot_ber_curve
    ebn0 = np.array([0.0, 5.0, 10.0, 20.0])
    ber = np.array([7.9e-2, 6e-3, 4e-6, 0.0])
    theory = ber_theory_curve('qpsk', np.linspace(0.0, 20.0, 100))
    curve = BerCurve(ebn0, ber, scheme='qpsk', n_bits=100000)
    fig, ax = curve.plot()
    theory.plot(ax=ax)
    try:
        assert [line.get_label() for line in ax.lines][-1] == 'qpsk theory'
        assert ax.get_ylim()[0] == pytest.approx(4e-7, rel=1e-9)
    finally:
        plt.close(fig)
    fig, ax = plot_ber_curve(ebn0, ber, n_bits=1e7)
    plot_ber_curve(theory, ax=ax)
    try:
        assert ax.get_ylim()[0] == pytest.approx(1e-8, rel=1e-9)
    finally:
        plt.close(fig)


def test_the_ber_axis_stops_at_a_rate_of_one():
    """A rate above 1 is not a value, and the log axis' own margin over a
    QPSK theory curve reaching 1e-45 puts the autoscaled top at 12.3. The
    top is 1 at most; a measured curve alone keeps its autoscaled top below
    it."""
    from uacpy.comms import ber_theory_curve
    from uacpy.plot import plot_ber_curve
    theory = ber_theory_curve('qpsk', np.linspace(0.0, 20.0, 100))
    ebn0 = np.array([0.0, 5.0, 10.0])
    measured = [9.98e-2, 6e-3, 1.9e-4]
    fig, ax = theory.plot()
    assert ax.get_ylim()[1] == 1.0
    plt.close(fig)
    fig, ax = plot_ber_curve(ebn0, measured)
    assert measured[0] < ax.get_ylim()[1] < 1.0
    theory.plot(ax=ax)
    assert ax.get_ylim()[1] == 1.0
    plt.close(fig)


@pytest.mark.parametrize('theory_first', [False, True])
def test_a_second_measured_curve_does_not_clip_the_first(theory_first):
    """The axis bound reads every measured curve on the axes: a second
    curve resolving only 1e-3 keeps the first curve's 1e-6 point in view
    (one decade below it), whichever order the curves and the theory line
    are drawn in."""
    from uacpy.comms import ber_theory_curve
    from uacpy.plot import plot_ber_curve
    ebn0 = np.array([0.0, 5.0, 10.0])
    theory = ber_theory_curve('qpsk', np.linspace(0.0, 20.0, 50))
    fig, ax = plt.subplots()
    calls = [lambda: plot_ber_curve(ebn0, [1e-1, 1e-3, 1e-6], ax=ax,
                                    label='deep'),
             lambda: plot_ber_curve(theory, ax=ax),
             lambda: plot_ber_curve(ebn0, [2e-1, 5e-2, 1e-3], ax=ax,
                                    label='shallow')]
    if theory_first:
        calls = [calls[1], calls[0], calls[2]]
    for call in calls:
        call()
    try:
        assert ax.get_ylim()[0] == pytest.approx(1e-7, rel=1e-9)
        assert ax.get_ylim()[1] >= 2e-1
    finally:
        plt.close(fig)


def test_a_theory_curve_is_a_record_it_draws_dashed():
    """``ber_theory_curve`` is ``ber_theory`` on the caller's grid; its
    ``.plot()`` is ``plot_ber_curve(curve)``, a dashed black line, and the
    plotter has no ``scheme=`` that would compute one."""
    from uacpy import comms
    from uacpy.plot import plot_ber_curve
    grid = np.linspace(0.0, 10.0, 11)
    theory = comms.ber_theory_curve('16qam', grid)
    np.testing.assert_array_equal(theory.ber, comms.ber_theory('16qam', grid))
    assert theory.scheme == '16qam'
    fig, ax = theory.plot()
    (line,) = ax.lines
    assert (line.get_linestyle(), line.get_color()) == ('--', 'k')
    assert line.get_label() == '16qam theory'
    plt.close(fig)
    assert 'scheme' not in inspect.signature(plot_ber_curve).parameters


def test_two_curves_on_one_axes_are_named_apart():
    """A ``BerCurve`` names its own legend entry from what its link carried,
    so two curves on one axes read apart. ``label=`` wins over it, and bare
    arrays are labelled 'measured'."""
    from uacpy.comms import BerCurve
    from uacpy.comms.equalize import DFE
    from uacpy.plot import plot_ber_curve
    ebn0 = np.array([0.0, 5.0, 10.0])
    ber = np.array([1e-1, 1e-2, 1e-3])
    awgn = BerCurve(ebn0, ber, scheme='qpsk', n_bits=1000)
    multipath = BerCurve(ebn0, ber, scheme='qpsk', n_bits=1000,
                         channel=np.array([1.0, 0.5]), equalizer=DFE())
    fig, ax = plot_ber_curve(awgn)
    plot_ber_curve(multipath, ax=ax)
    plot_ber_curve(ebn0, ber, ax=ax)
    plot_ber_curve(ebn0, ber, ax=ax, label='mine')
    try:
        labels = [line.get_label() for line in ax.lines
                  if 'theory' not in line.get_label()]
        assert labels == ['qpsk', 'qpsk, channel, DFE', 'measured', 'mine']
    finally:
        plt.close(fig)


class TestOneNullPolicyForMagnitudesInDecibels:
    """A null of ``20·log10|x|`` is drawn on a floor ``dynamic_range_dB``
    below the peak (120 dB by default), by every magnitude-in-dB plotter,
    with no numpy warning and no point dropped."""

    def test_the_channel_and_subcarrier_views_floor_a_null_alike(self):
        import warnings
        import matplotlib.pyplot as plt
        from uacpy.acoustic_signal import channel_response
        from uacpy.plot import plot_subcarriers
        taps = np.array([1.0, 1.0])              # H = 0 at Nyquist
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fig, ax_f = channel_response(taps, 2.0).plot()
            channel = ax_f.lines[-1].get_ydata()
            fig2, ax_s = plot_subcarriers(np.fft.fft(taps, 8))
            subcarriers = ax_s.lines[-1].get_ydata()
        for drawn in (channel, subcarriers):
            assert np.all(np.isfinite(drawn))
            assert np.min(drawn) == pytest.approx(np.max(drawn) - 120.0)
        plt.close('all')

    @pytest.mark.parametrize('span', [20.0, 60.0])
    def test_the_floor_follows_dynamic_range_dB(self, span):
        import matplotlib.pyplot as plt
        from uacpy.plot import plot_subcarriers
        fig, ax = plot_subcarriers(np.fft.fft([1.0, 1.0], 8),
                                   dynamic_range_dB=span)
        drawn = ax.lines[-1].get_ydata()
        assert np.min(drawn) == pytest.approx(np.max(drawn) - span)
        plt.close(fig)
