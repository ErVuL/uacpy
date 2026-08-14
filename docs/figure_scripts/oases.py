"""Figures for ``docs/models/oases.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import SQUARE, WIDE, layered_elastic, shallow_water

import uacpy
from uacpy.models import OAST, OASN, OASP, OASR, RunMode


def elastic_stack():
    """The shared ``layered_elastic`` seabed with its shear speeds kept.

    ``from_presets`` drops shear by default so every model can consume the
    column. OASES is one of the few that does not need that concession.
    """
    return uacpy.Environment(
        name='Sand over granite (elastic)',
        bathymetry=100.0,
        ssp=[(0.0, 1500.0), (100.0, 1490.0)],
        bottom=uacpy.SeabedColumn.from_presets(
            layers=[('sand', 8.0)], halfspace='granite', elastic=True,
        ),
    )


def rock_seabed(*, elastic: bool):
    """Bare granite under the water column, with and without shear."""
    return uacpy.Environment(
        name='Rock seabed' + (' (elastic)' if elastic else ' (fluid)'),
        bathymetry=100.0,
        ssp=[(0.0, 1500.0), (100.0, 1490.0)],
        bottom=uacpy.BoundaryProperties.from_preset('granite', elastic=elastic),
    )


def oast_tl():
    """OAST transmission loss over two seabeds, same water column.

    Top: the shared shallow-water channel over a fluid sand half-space, the
    same case the Bellhop and Kraken pages draw. Bottom: an 8 m sand layer
    over an elastic granite basement, which only OASES consumes exactly.
    """
    env_fluid, source, receiver = shallow_water()
    env_elastic = elastic_stack()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, sharey=True)
    for ax, env, label in ((axes[0], env_fluid, 'Fluid sand half-space'),
                           (axes[1], env_elastic,
                            'Sand layer over elastic granite')):
        tl = OAST().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        tl.plot(env=env, source=source, ax=ax,
                show_colorbar=(ax is axes[0]))
        ax.set_title(label, fontweight='bold', fontsize=11)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('OAST — coherent transmission loss, 200 Hz',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def _range_average(tl_db, n=15):
    """Average TL in intensity over ``n`` range cells, returning dB."""
    kernel = np.ones(n) / n
    intensity = np.convolve(10.0 ** (-tl_db / 10.0), kernel, mode='valid')
    return -10.0 * np.log10(intensity)


def shear_loss():
    """What shear costs: the same rock seabed as a fluid and as a solid.

    Below the critical angle a fluid rock half-space reflects perfectly, so
    the only loss is cylindrical spreading. Shear conversion leaks a little
    energy at every bounce, and over 10 km of multipath that compounds. The
    TL cuts are averaged in intensity over 500 m so the trend, rather than
    the interference pattern, is what you read.
    """
    source = uacpy.Source(depths=25.0, frequencies=200.0)
    receiver = uacpy.Receiver(depths=50.0,
                              ranges=np.linspace(50.0, 10_000.0, 300))
    angles = np.linspace(0.0, 90.0, 181)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for elastic, colour, label in ((False, 'C0', 'Fluid (shear dropped)'),
                                   (True, 'C3', 'Elastic (shear kept)')):
        env = rock_seabed(elastic=elastic)
        rc = OASR(angles=angles).run(env, source, receiver)
        axes[0].plot(rc.theta, -20.0 * np.log10(np.maximum(rc.R, 1e-6)),
                     color=colour, label=label)
        tl = OAST().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        smoothed = _range_average(tl.db.ravel())
        ranges = tl.coords['range'][:len(smoothed)] + 250.0
        axes[1].plot(ranges / 1000.0, smoothed, color=colour, label=label)

    axes[0].set_xlabel('Grazing angle (°)')
    axes[0].set_ylabel('Reflection loss (dB per bounce)')
    axes[0].set_title('OASR — bottom loss', fontweight='bold', fontsize=11)
    axes[0].set_ylim(0.0, 3.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('Transmission loss (dB)')
    axes[1].set_title('OAST — TL at 50 m depth', fontweight='bold', fontsize=11)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle('OASES — a rock seabed with and without shear, 200 Hz',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def oasr_reflection():
    """Reflection coefficient of the layered seabed, fluid versus elastic."""
    env_fluid, source, receiver = layered_elastic()
    angles = np.linspace(0.0, 90.0, 181)
    cases = ((env_fluid, 'Shear dropped (fluid stack)'),
             (elastic_stack(), 'Shear kept (elastic stack)'))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (env, label) in zip(axes, cases):
        rc = OASR(angles=angles).run(env, source, receiver,
                                     run_mode=RunMode.REFLECTION)
        rc.plot(ax=ax, show_phase=True, title=label)
        ax.set_ylim(0.0, 1.05)
    fig.suptitle('OASR — plane-wave reflection off sand over granite, 200 Hz',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def oasr_broadband():
    """|R(θ, f)| over the elastic stack — the sand layer's resonances."""
    _, source, receiver = layered_elastic()
    rc = OASR(angles=np.linspace(0.0, 90.0, 181)).run(
        elastic_stack(), source, receiver, run_mode=RunMode.REFLECTION,
        frequencies=np.linspace(20.0, 2000.0, 120))
    fig, ax = rc.plot(
        figsize=WIDE,
        title='OASR — |R(θ, f)| for an 8 m sand layer over elastic granite')
    return fig


def oasp_gather():
    """An OASP pulse gather: p(t) against range at a near-bottom receiver."""
    env = elastic_stack()
    source = uacpy.Source(depths=25.0, frequencies=100.0)
    receiver = uacpy.Receiver(depths=90.0,
                              ranges=np.linspace(100.0, 1500.0, 96))

    sample_rate = 800.0
    t = np.arange(96) / sample_rate
    t0 = t[len(t) // 2]
    pulse = (np.sin(2 * np.pi * 100.0 * (t - t0))
             * np.exp(-((t - t0) ** 2) / (2 * 0.012 ** 2)))

    ts = OASP(n_time_samples=512, freq_max=200.0).run(
        env, source, receiver, run_mode=RunMode.TIME_SERIES,
        source_waveform=pulse, sample_rate=sample_rate, output_duration=1.2)
    gather = ts.at(depth=90.0)
    peak = float(np.max(np.abs(gather.data)))
    fig, ax = gather.plot(
        figsize=WIDE, vmin=-0.25 * peak, vmax=0.25 * peak,
        title='OASP — pulse gather at 90 m depth, 100 Hz centre frequency')
    return fig


def oasn_covariance():
    """Surface-generated noise as seen by a 24-element vertical array."""
    env = elastic_stack()
    source = uacpy.Source(depths=40.0, frequencies=100.0)
    array = uacpy.Receiver(depths=np.linspace(20.0, 90.0, 24), ranges=0.0)

    cov = OASN(surface_noise_level=60.0,
               white_noise_level=20.0).compute_covariance(env, source, array)
    fig, ax = cov.plot(figsize=SQUARE,
                       title='OASN — surface-noise |C|, 24-element VLA')
    return fig


def oasn_mfp():
    """Matched-field localisation from OASN replicas and covariance.

    One OASN run gives the measured covariance of a discrete source in
    white noise; a second gives the replica field over a candidate grid.
    Contracting them is matched-field processing.
    """
    env = elastic_stack()
    source = uacpy.Source(depths=40.0, frequencies=100.0)
    array = uacpy.Receiver(depths=np.linspace(20.0, 90.0, 24), ranges=0.0)

    cov = OASN(
        white_noise_level=40.0,
        discrete_sources=[{'depth': 40.0, 'x': 3000.0, 'y': 0.0,
                           'level': 100.0}],
    ).compute_covariance(env, source, array)
    replicas = OASN(
        xmin=500.0, xmax=6000.0, nx=111,
        zmin=5.0, zmax=95.0, nz=46,
    ).compute_replicas(env, source, array)

    surfaces = ((cov.bartlett(replicas)[0, :, :, 0], 'Bartlett'),
                (cov.mvdr(replicas)[0, :, :, 0], 'MVDR (Capon)'))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (surface, label) in zip(axes, surfaces):
        im = ax.pcolormesh(replicas.replica_x / 1000.0, replicas.replica_z,
                           10.0 * np.log10(surface / surface.max()),
                           shading='nearest', cmap='inferno',
                           vmin=-15.0, vmax=0.0)
        ax.plot(3.0, 40.0, marker='*', ms=15, mfc='none', mec='cyan', mew=1.5)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_xlabel('Range (km)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].invert_yaxis()
    fig.colorbar(im, ax=axes, label='Ambiguity power (dB re peak)')
    fig.suptitle('OASN — matched-field ambiguity surfaces; ☆ is the true '
                 'source', fontweight='bold', fontsize=13)
    return fig


FIGURES = {
    'oases_oast_tl': oast_tl,
    'oases_shear_loss': shear_loss,
    'oases_oasr_reflection': oasr_reflection,
    'oases_oasr_broadband': oasr_broadband,
    'oases_oasp_gather': oasp_gather,
    'oases_oasn_covariance': oasn_covariance,
    'oases_oasn_mfp': oasn_mfp,
}
