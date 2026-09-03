"""Figures for ``docs/models/sparc.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

SPARC marches a pulse, so every run costs real time. The receivers below are
kept deliberately sparse and ``t_max`` is always pinned — the auto window
(2.5 x the travel time to ``RMax``) is far longer than these figures need.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import TALL, WIDE, shallow_water

import uacpy
from uacpy.core.results import Field
from uacpy.models import SPARC
from uacpy.visualization import plot_time_snapshots


def record_section():
    """``output_mode='R'``: a horizontal array, drawn as a record section.

    Traces are scaled by ``sqrt(r)`` to undo cylindrical spreading — the
    standard record-section gain, without which the near traces swamp the
    far ones and the moveout is invisible.
    """
    env, source, _ = shallow_water()
    array = uacpy.Receiver(depths=50.0, ranges=np.linspace(200.0, 1200.0, 24))
    p = SPARC(t_max=1.0, n_t_out=1024).run(env, source, array)

    section = p.at(depth=50.0).to_dict()
    section['data'] = (section['data']
                       * np.sqrt(section['coords']['range'])[:, None])
    fig, ax = Field.from_dict(section).plot(
        stacked=True, figsize=TALL,
        title='SPARC — horizontal array at 50 m, gained by √r')
    # km, not m: the stacked view reads a range axis in km, as every
    # other view of a range axis does. The override said metres over
    # ticks reading 0.2 to 1.157 for an array spanning 200 to 1200 m.
    ax.set_ylabel('Range (km, stacked)')
    return fig


def single_trace():
    """One receiver: the pulse that arrives, and the band it occupies."""
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=50.0, ranges=np.array([800.0]))
    p = SPARC(t_max=0.8, n_t_out=2048).run(env, source, point)
    trace = p.at(depth=50.0, range=800.0)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0))
    trace.plot(ax=axes[0])
    axes[0].set_title('p(t) at (800 m, 50 m)', fontweight='bold', fontsize=11)

    freqs, spectrum = trace.get_spectrum()
    axes[1].semilogy(freqs, np.abs(spectrum) + 1e-30, linewidth=1.0)
    for edge, label in ((100.0, 'f_min = fc/2'), (400.0, 'f_max = 2·fc')):
        axes[1].axvline(edge, color='crimson', linestyle='--', linewidth=1.0)
        axes[1].annotate(label, (edge, 1.0), xycoords=('data', 'axes fraction'),
                         xytext=(4, -12), textcoords='offset points',
                         color='crimson', fontsize=9)
    axes[1].set_xlim(0.0, 700.0)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('|P(f)|')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('and its spectrum — f_min / f_max bound the wavenumbers '
                      'marched', fontweight='bold', fontsize=11)
    fig.suptitle('SPARC — the received pulse and the pulse band',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def vertical_array():
    """``output_mode='D'``: the same event down a vertical array.

    One run per receiver *range* instead of per depth, so a whole VLA at a
    single range costs one SPARC run.
    """
    env, source, _ = shallow_water()
    vla = uacpy.Receiver(depths=np.linspace(2.0, 98.0, 33),
                         ranges=np.array([800.0]))
    p = SPARC(output_mode='D', t_max=0.9, n_t_out=1024).run(env, source, vla)
    fig, ax = p.at(range=800.0).plot(
        stacked=True, figsize=TALL,
        title='SPARC — vertical array at 800 m (output_mode=\'D\')')
    ax.set_ylabel('Depth (m, stacked)')
    return fig


def snapshots():
    """``output_mode='S'``: the wavefront itself, at four instants."""
    env, source, _ = shallow_water()
    grid = uacpy.Receiver(depths=np.linspace(1.0, 99.0, 40),
                          ranges=np.linspace(10.0, 500.0, 80))
    p = SPARC(output_mode='S', t_max=0.36, n_t_out=384).run(env, source, grid)

    times = (0.06, 0.14, 0.22, 0.30)
    late = np.asarray(p.data)[..., -1]
    fig, _ = plot_time_snapshots(
        {'SPARC': p}, times, env=env,
        p_max=float(np.percentile(np.abs(late[np.isfinite(late)]), 99.0)),
        title='SPARC — pulse marching down the channel (output_mode=\'S\')')
    return fig


def pulse_shapes():
    """``pulse_type`` position 1 picks the source waveform.

    The pseudo-Gaussian and the Ricker wavelet are impulsive; the
    Hanning-weighted four-sine is a narrowband tone burst, and the received
    field rings for as long as the source did.
    """
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=50.0, ranges=np.array([600.0]))
    pulses = [('PN+B', 'Pseudo-Gaussian, band-passed (default)'),
              ('RN+N', 'Ricker wavelet'),
              ('HN+N', 'Hanning-weighted four sine')]

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for ax, (code, label) in zip(axes, pulses):
        p = SPARC(pulse_type=code, t_max=0.7, n_t_out=2048).run(
            env, source, point)
        p.at(depth=50.0, range=600.0).plot(ax=ax)
        ax.set_title(f"pulse_type='{code}' — {label}",
                     fontweight='bold', fontsize=11)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('SPARC — three source pulses, same channel and receiver',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def output_sampling():
    """``n_t_out`` is the output Nyquist, and SPARC says so when it is short.

    The window is ``[0, t_max]`` inclusive of both ends, so the output sample
    rate is ``(n_t_out - 1) / t_max`` — the same expression the wrapper's own
    aliasing check uses (``SPARC._resolve_n_t_out``), not ``n_t_out / t_max``,
    which overstates it by ``n/(n-1)``. Below ``2·f_max`` the marched pulse is
    folded back into the band and p(t) looks plausible at the wrong frequency.
    """
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=50.0, ranges=np.array([600.0]))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=True)
    for ax, n_t_out in zip(axes, (128, 2048)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            p = SPARC(t_max=0.7, n_t_out=n_t_out).run(env, source, point)
        p.at(depth=50.0, range=600.0).plot(ax=ax)
        fs = (n_t_out - 1) / 0.7
        ax.set_title(f'n_t_out={n_t_out} → {fs:.0f} Hz sampling, '
                     f'Nyquist {fs / 2:.0f} Hz',
                     fontweight='bold', fontsize=11)
        if any('will alias' in str(w.message) for w in caught):
            ax.text(0.99, 0.05, 'UserWarning: p(t) will alias', ha='right',
                    va='bottom', transform=ax.transAxes, fontsize=10,
                    color='crimson', fontweight='bold')
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('SPARC — the output grid must resolve the pulse band',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'sparc_record_section': record_section,
    'sparc_trace': single_trace,
    'sparc_vertical_array': vertical_array,
    'sparc_snapshots': snapshots,
    'sparc_pulse_shapes': pulse_shapes,
    'sparc_output_sampling': output_sampling,
}
