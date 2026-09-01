"""Pins for the raw-array comms plotters in ``plots.comms``.

``plot_subcarriers``, ``plot_channel`` and ``plot_eye_diagram`` carry real
axis/indexing contracts (OFDM bin addressing, ms delay axis, 0 Hz-centred
spectrum, symbol-interval windowing) and are asserted on the drawn data; the
thin wrappers (``plot_convergence``, ``plot_sync_metric``,
``plot_doppler_ambiguity``) get their one transform pinned plus a shared
smoke/``ax=`` sweep. The comms numerics themselves live in ``test_comms.py``
and ``test_equalization.py``; the constellation/BER plotters are covered in
``test_carrier_plot_methods.py`` callers and the doc figure gates.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from uacpy.visualization import (
    plot_channel, plot_convergence, plot_doppler_ambiguity, plot_eye_diagram,
    plot_subcarriers, plot_sync_metric,
)

# Asymmetric complex taps: |H(f)| has no conjugate symmetry, so any
# fftshift/index-origin slip moves values, not just labels.
_TAPS = np.array([1.0, 0.5j, -0.25])


def test_plot_subcarriers_addresses_bins_like_ofdm_demodulate():
    # ofdm_demodulate equalizes bin k with np.fft.fft(h, nsc)[k]; the plot
    # must put that same value at x = k (unshifted, DC at index 0).
    nsc = 8
    fig, ax = plot_subcarriers(_TAPS, nsc)
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), np.arange(nsc))
    expected = 20 * np.log10(np.abs(np.fft.fft(_TAPS, nsc)) + 1e-12)
    np.testing.assert_allclose(line.get_ydata(), expected)
    plt.close(fig)


def test_plot_channel_builds_its_own_two_panel_figure_when_ax_is_none():
    fig, ax = plot_channel(_TAPS, 1000.0)
    assert len(ax) == 2
    assert ax[0].figure is fig and ax[1].figure is fig
    assert ax[0].containers          # stem in the delay panel
    assert ax[1].lines               # spectrum in the frequency panel
    plt.close(fig)


def test_plot_channel_draws_on_a_given_tuple_of_axes():
    fig, (ax_d, ax_f) = plt.subplots(1, 2)
    out_fig, out_ax = plot_channel(_TAPS, 1000.0, (ax_d, ax_f))
    assert out_fig is fig
    assert out_ax[0] is ax_d and out_ax[1] is ax_f
    assert ax_d.containers and ax_f.lines
    plt.close(fig)


def test_plot_channel_delay_axis_is_milliseconds():
    fig, ax = plot_channel(np.array([1.0, 0.5, 0.25, 0.1]), 1000.0)
    stem_x = ax[0].containers[0].markerline.get_xdata()
    np.testing.assert_allclose(stem_x, [0.0, 1.0, 2.0, 3.0])
    plt.close(fig)


def test_plot_channel_spectrum_is_centred_on_zero_hz():
    fs = 500.0
    fig, ax = plot_channel(_TAPS, fs)
    f = ax[1].lines[0].get_xdata()
    mag_db = ax[1].lines[0].get_ydata()
    assert np.all(np.diff(f) > 0)
    assert f[0] == -fs / 2
    # H(0) = sum of the taps; misaligned f/H shifts would break this pairing.
    dc = 20 * np.log10(np.abs(_TAPS.sum()) + 1e-12)
    np.testing.assert_allclose(mag_db[f == 0.0], [dc])
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


def test_plot_doppler_ambiguity_plots_scales_in_thousandths_and_marks_the_peak():
    scales = np.linspace(-2e-3, 2e-3, 5)
    peak = np.array([0.1, 0.2, 0.3, 1.0, 0.4])
    fig, ax = plot_doppler_ambiguity(scales, peak)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), scales * 1e3)
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 1.0])
    plt.close(fig)


def test_plot_convergence_plots_mse_in_decibels_with_a_floor():
    mse = np.array([1.0, 0.1, 0.01, 0.0])
    fig, ax = plot_convergence(mse)
    np.testing.assert_allclose(ax.lines[0].get_ydata(),
                               [0.0, -10.0, -20.0, -120.0])
    plt.close(fig)


@pytest.mark.parametrize("plotter, args", [
    (plot_convergence, (np.array([1.0, 0.5, 0.1]),)),
    (plot_sync_metric, (np.array([0.1, 0.8, 0.2]),)),
    (plot_doppler_ambiguity, (np.linspace(-1e-3, 1e-3, 5),
                              np.array([0.2, 0.4, 1.0, 0.5, 0.3]))),
    (plot_subcarriers, (_TAPS, 8)),
])
def test_comms_plotters_return_fig_ax_and_honour_ax(plotter, args):
    fig, ax = plotter(*args)
    assert fig is not None and ax.figure is fig
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    _, ax3 = plotter(*args, ax=ax2)
    assert ax3 is ax2
    plt.close(fig2)
