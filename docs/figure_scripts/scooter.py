"""Figures for ``docs/models/scooter.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import (
    WIDE, layered_elastic, shallow_water, sloping_shelf,
)

import uacpy
from uacpy.io import read_grn_file
from uacpy.models import Bellhop, Kraken, RunMode, Scooter


def run_with_greens_function(env, source, receiver, **knobs):
    """Run Scooter keeping its scratch dir, and read the ``.grn`` back.

    Pinning ``work_dir`` switches cleanup off, so the Green's function
    survives the run and its path lands in ``field.metadata['grn_file']``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        field = Scooter(work_dir=Path(tmp), **knobs).run(env, source, receiver)
        return field, read_grn_file(field.metadata['grn_file'])


def _wavenumbers(grn):
    """Horizontal wavenumbers behind a ``.grn``: ``k = 2πf / c``."""
    return 2.0 * np.pi * grn['freq'] / grn['cVec']


def tl_field():
    """Coherent TL over the shallow-water channel, with the seabed drawn."""
    env, source, receiver = shallow_water()
    tl = Scooter().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    fig, ax = tl.plot(env=env, source=source, figsize=WIDE,
                      title='Scooter — coherent transmission loss, 200 Hz')
    return fig


def greens_function():
    """The wavenumber-domain field Scooter solves before the transform.

    Top: ``|G(k, z)|``, the depth-separated Green's function on the
    horizontal-wavenumber axis. Bottom: a cut at the source depth. Every
    spike marks a pole of ``G`` — those inside the shaded band are the
    trapped modes — and Kraken's eigenvalues, found by an entirely
    different algorithm, land on them.
    """
    env, source, receiver = shallow_water()
    _, grn = run_with_greens_function(env, source, receiver)
    k = _wavenumbers(grn)
    G = np.abs(grn['G'][0, 0])                       # (n_rd, n_k)
    G_db = 20.0 * np.log10(G / G.max() + 1e-12)
    modes = Kraken().compute_modes(env, source)

    freq = float(grn['freq'])
    c_water = float(np.min(env.ssp.to_pairs()[:, 1]))
    c_bottom = float(env.bottom.halfspace_at(range=0.0).sound_speed)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1.0]})
    im = axes[0].imshow(
        G_db, aspect='auto', origin='upper', cmap='inferno', vmin=-55, vmax=0,
        extent=[k[0], k[-1], grn['rd'][-1], grn['rd'][0]])
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title(f"$|G(k, z)|$ — {grn['nk']} wavenumbers × "
                      f"{grn['nrd']} depths", fontweight='bold', fontsize=11)
    fig.colorbar(im, ax=axes[0], pad=0.02, label='dB re max')
    top = axes[0].secondary_xaxis(
        'top', functions=(lambda x: 2.0 * np.pi * freq / np.maximum(x, 1e-9),
                          lambda c: 2.0 * np.pi * freq / np.maximum(c, 1e-9)))
    top.set_xlabel('Phase speed $c = \\omega/k$ (m/s)')

    zi = int(np.argmin(np.abs(grn['rd'] - float(source.depths[0]))))
    k_bottom = 2.0 * np.pi * freq / c_bottom
    k_water = 2.0 * np.pi * freq / c_water
    axes[1].axvspan(k_bottom, k_water, color='0.85', zorder=0,
                    label=f'trapped: {c_bottom:.0f} > c > {c_water:.0f} m/s')
    axes[1].plot(k, G_db[zi], color='C0', linewidth=0.9,
                 label=f"$|G(k)|$ at z = {grn['rd'][zi]:.0f} m")
    for i, km in enumerate(np.real(modes.k)):
        axes[1].axvline(km, color='C3', linestyle=':', linewidth=1.0,
                        label='Kraken eigenvalues $k_m$' if i == 0 else None)
    for ax, colour in zip(axes, ('white', '0.25')):
        for kc in (k_bottom, k_water):
            ax.axvline(kc, color=colour, linestyle='--', linewidth=1.2)
    axes[1].set_xlim(k[0], k[-1])
    axes[1].set_ylim(-60, 3)
    axes[1].set_xlabel('Horizontal wavenumber $k$ (rad/m)')
    axes[1].set_ylabel('dB re max')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Scooter — the Green\u2019s function before the Hankel '
                 'transform, 200 Hz', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def phase_speed_window():
    """What ``c_low`` / ``c_high`` include in the wavenumber integral.

    The window is the integration interval: ``k`` runs from ``ω/c_high`` to
    ``ω/c_low``, and everything outside it is simply not computed. Raising
    ``c_low`` truncates the high-``k`` end (the best-trapped modes);
    lowering ``c_high`` truncates the low-``k`` end (near-cutoff modes and
    the continuum that leaks into the seabed).
    """
    env, source, _ = shallow_water()
    line = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 400))
    cases = [('default — 1416 to 1732 m/s', {}, 'C0'),
             ('c_high=1600', {'c_high': 1600.0}, 'C1'),
             ('c_low=1520', {'c_low': 1520.0}, 'C2')]
    freq = float(source.frequencies[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for label, knobs, colour in cases:
        tl, grn = run_with_greens_function(env, source, line, **knobs)
        k = _wavenumbers(grn)
        if not knobs:
            G = np.abs(grn['G'][0, 0, 0])
            axes[0].plot(k, 20.0 * np.log10(G / G.max() + 1e-12), color=colour,
                         linewidth=0.9, zorder=3)
            k_full = k
        else:
            span = ((k_full[0], k[0]) if 'c_high' in knobs
                    else (k[-1], k_full[-1]))
            axes[0].axvspan(*span, color=colour, alpha=0.22, zorder=1,
                            label=f'excluded by {label}')
        axes[1].plot(np.asarray(line.ranges) / 1000.0,
                     np.asarray(tl.db, dtype=float).ravel(),
                     color=colour, linewidth=1.0,
                     label=f"{label} — {grn['nk']} k-samples")

    axes[0].set_xlim(k_full[0], k_full[-1])
    axes[0].set_xlabel('Horizontal wavenumber $k$ (rad/m)')
    axes[0].set_ylabel('$|G(k)|$ at 50 m (dB re max)')
    axes[0].set_title('What each window leaves out', fontweight='bold',
                      fontsize=11)
    axes[0].set_ylim(-60, 3)
    axes[0].legend(fontsize=8, loc='lower center')
    axes[0].grid(True, alpha=0.3)
    top = axes[0].secondary_xaxis(
        'top', functions=(lambda x: 2.0 * np.pi * freq / np.maximum(x, 1e-9),
                          lambda c: 2.0 * np.pi * freq / np.maximum(c, 1e-9)))
    top.set_xlabel('Phase speed $c = \\omega/k$ (m/s)')

    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('TL (dB)')
    axes[1].set_title('Resulting TL at 50 m depth', fontweight='bold',
                      fontsize=11)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Scooter — the phase-speed window bounds the wavenumber '
                 'integral', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def elastic_seabed():
    """A layered seabed, and what turning its shear on costs.

    Top: the shared 8 m-sand-over-granite column, layer stack intact —
    Scooter meshes the sediment as a second medium instead of collapsing
    it. Bottom: the same presets rebuilt with ``elastic=True``, so the
    granite carries a 3000 m/s shear speed and the sand a slow one. Shear
    opens a loss channel no fluid seabed has.
    """
    env, source, receiver = layered_elastic()
    tl = Scooter().run(env, source, receiver)
    elastic = uacpy.Environment(
        name='Layered seabed, shear on',
        bathymetry=env.bathymetry, ssp=env.ssp,
        bottom=uacpy.SeabedColumn.from_presets(
            layers=[('sand', 8.0)], halfspace='granite', elastic=True),
    )
    line = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 400))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8),
                             gridspec_kw={'height_ratios': [1.5, 1.0]})
    tl.plot(env=env, source=source, ax=axes[0], show_colorbar=True)
    axes[0].set_title('8 m sand layer over a granite basement, 200 Hz',
                      fontweight='bold', fontsize=11)
    for label, e, colour in (('fluid seabed (no shear)', env, 'C0'),
                             ('elastic=True — $c_s$ = 3000 m/s in the '
                              'granite', elastic, 'C3')):
        line_tl = np.asarray(Scooter().run(e, source, line).db,
                             dtype=float).ravel()
        axes[1].plot(np.asarray(line.ranges) / 1000.0, line_tl,
                     color=colour, linewidth=1.0, label=label)
    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('TL (dB)')
    axes[1].set_title('TL at 50 m depth, with and without shear',
                      fontweight='bold', fontsize=11)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('Scooter — layered and elastic seabeds, meshed directly',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def benchmark():
    """Scooter as the reference: two approximations measured against it.

    Kraken discards the continuous spectrum and keeps a finite mode sum;
    Bellhop replaces the wave equation with rays. Scooter does neither, so
    the gap to it is the error each approximation costs.
    """
    env, source, _ = shallow_water()
    line = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 400))
    reference = np.asarray(Scooter().run(env, source, line).db,
                           dtype=float).ravel()
    others = [('Kraken (normal modes)', Kraken(), 'C1'),
              ('Bellhop (Gaussian beams)', Bellhop(n_beams=3000), 'C2')]

    r_km = np.asarray(line.ranges) / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.0]})
    axes[0].plot(r_km, reference, color='C0', linewidth=1.4,
                 label='Scooter (wavenumber integration)')
    for label, model, colour in others:
        tl = np.asarray(model.run(env, source, line).db, dtype=float).ravel()
        axes[0].plot(r_km, tl, color=colour, linewidth=0.9, alpha=0.85,
                     label=label)
        axes[1].plot(r_km, np.abs(tl - reference), color=colour,
                     linewidth=0.9, label=label)

    axes[0].set_ylabel('TL (dB)')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Transmission loss at 50 m depth', fontweight='bold',
                      fontsize=11)

    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('|Δ TL| (dB)')
    axes[1].set_title('Departure from the reference', fontweight='bold',
                      fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Scooter — the answer the other models are checked against, '
                 '200 Hz', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def range_collapse():
    """What a stratified solver does with a sloping shelf.

    Scooter integrates over one horizontal wavenumber axis, which only
    exists if the medium is stratified. uacpy collapses the bathymetry to a
    single depth and warns; the panels are the price of that collapse.
    """
    env, source, receiver = sloping_shelf()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flat = Scooter().run(env, source, receiver)
        sloped = Bellhop(n_beams=3000).run(env, source, receiver)
    bathy = env.bathymetry.to_pairs()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True,
                             sharey=True)
    flat.plot(ax=axes[0], source=source, show_colorbar=True)
    axes[0].plot(bathy[:, 0] / 1000.0, bathy[:, 1], color='k',
                 linestyle='--', linewidth=1.6)
    axes[0].set_title(f"Scooter — flat {env.depth:.0f} m "
                      "(collapse method='max'); dashed = the real seafloor",
                      fontweight='bold', fontsize=11)
    axes[0].set_xlabel('')

    sloped.plot(env=env, source=source, ax=axes[1], show_colorbar=False)
    axes[1].set_title('Bellhop — the same shelf, range dependence kept',
                      fontweight='bold', fontsize=11)

    fig.suptitle('Scooter — range dependence is collapsed, not modelled, '
                 '100 Hz', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def broadband():
    """Exact ``H(f)``, one full wavenumber solve per frequency.

    Bellhop synthesises the same band from a single arrivals run by phasing
    ray delays. Where the two agree, the ray approximation is safe here;
    where they part, it is not.
    """
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.linspace(150.0, 450.0, 128))
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H_exact = Scooter().run(env, source, point, run_mode=RunMode.BROADBAND)
        H_rays = Bellhop(n_beams=3000).run(env, source, point,
                                           run_mode=RunMode.BROADBAND)

    fig, ax = plt.subplots(figsize=WIDE)
    for H, label, colour, width in (
            (H_exact, 'Scooter — 128 wavenumber solves', 'C0', 1.4),
            (H_rays, 'Bellhop — one arrivals run, phased', 'C2', 1.0)):
        d = H.to_dict()
        ax.plot(d['coords']['frequency'],
                20.0 * np.log10(np.abs(d['data']).ravel() + 1e-30),
                color=colour, linewidth=width, label=label)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$20\\log_{10}|H|$ (dB)')
    ax.set_title('Scooter — broadband transfer function at (3 km, 60 m)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


FIGURES = {
    'scooter_tl': tl_field,
    'scooter_greens_function': greens_function,
    'scooter_phase_speed_window': phase_speed_window,
    'scooter_elastic': elastic_seabed,
    'scooter_benchmark': benchmark,
    'scooter_range_collapse': range_collapse,
    'scooter_broadband': broadband,
}
