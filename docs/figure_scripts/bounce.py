"""Figures for ``docs/models/bounce.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

Every reflection figure puts the same isovelocity water column over a different
seabed, so critical angles and plateau heights are directly comparable from one
figure to the next — the bottom is the only thing that changes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import TALL, WIDE, layered_elastic

import uacpy
from uacpy.models import Bellhop, Bounce, RunMode

# One angular grid for every figure. ``c_high=1e9`` zeroes BOUNCE's minimum
# wavenumber, which is what buys coverage all the way down to 0° grazing; a
# pinned ``rmax`` makes the sample count independent of the receiver.
GRID = dict(c_low=1400.0, c_high=1e9, rmax=10_000.0)

# BOUNCE reads no source or receiver geometry. A Receiver is still required by
# the uniform ``run()`` signature — it is consulted only for ``range_max``,
# which ``rmax`` above has already pinned.
PROBE = uacpy.Receiver(depths=50.0, ranges=10_000.0)

WATER_SPEED = 1500.0


def seabed(bottom):
    """100 m of isovelocity water over ``bottom`` — the one thing that varies."""
    return uacpy.Environment(
        name='Reflection test case',
        bathymetry=100.0,
        ssp=[(0.0, WATER_SPEED), (100.0, WATER_SPEED)],
        bottom=bottom,
    )


def reflection(bottom, frequency=200.0):
    """Plane-wave reflection coefficient of ``bottom`` at ``frequency``."""
    source = uacpy.Source(depths=25.0, frequencies=frequency)
    return Bounce(**GRID).run(seabed(bottom), source, PROBE)


def critical_angle(sound_speed):
    """Grazing angle below which a wave in the water cannot enter ``sound_speed``."""
    return np.degrees(np.arccos(WATER_SPEED / sound_speed))


def as_explicit_layer(bottom, thickness=1.0):
    """``bottom`` rewritten as one explicit layer of itself over itself.

    A bare half-space makes uacpy hand BOUNCE a dummy slab at the *water* speed,
    which leaves the tabulated phase referenced one metre above the seafloor —
    see the Gotchas section of ``docs/models/bounce.md``. An explicit layer sends
    the same seabed down BOUNCE's exact path instead. The layer is the same
    material as the half-space, so the seabed is physically unchanged and ``|R|``
    is identical; only the phase reference moves.

    Fluid media only: BOUNCE marches acoustic layers, so an elastic layer is not
    a valid way to spell an elastic half-space.
    """
    return uacpy.SeabedColumn(
        layers=[uacpy.SedimentLayer(
            thickness=thickness,
            sound_speed=bottom.sound_speed,
            density=bottom.density,
            attenuation=bottom.attenuation,
        )],
        halfspace=bottom,
    )


def reflection_coefficient():
    """The canonical one-liner: |R(θ)| and its phase, from ``.plot()``."""
    sand = uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1650.0, density=1.9, attenuation=0.0,
    )
    rc = reflection(as_explicit_layer(sand))
    fig, ax = rc.plot(
        show_phase=True, figsize=WIDE,
        title='Bounce — lossless sand half-space, 200 Hz')
    return fig


def attenuation_ladder():
    """What compressional attenuation does to total internal reflection.

    Below the critical angle a lossless seabed returns every bit of energy.
    Add absorption and the plateau sags — that sag is the per-bounce loss a
    waveguide accumulates over range.
    """
    fig, ax = plt.subplots(figsize=WIDE)
    for alpha, style in [(0.0, '-'), (0.2, '--'), (0.6, '-.'), (1.5, ':')]:
        rc = reflection(uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1650.0, density=1.9, attenuation=alpha,
        ))
        ax.plot(rc.theta, rc.R, style, lw=1.6,
                label=f'α = {alpha:g} dB/λ')
    theta_c = critical_angle(1650.0)
    ax.axvline(theta_c, color='0.35', lw=1.0,
               label=f'critical angle {theta_c:.1f}°')
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Grazing angle (°)')
    ax.set_ylabel('|R|')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title('Bounce — sand half-space, 1650 m/s, varying attenuation',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig


def material_presets():
    """Six seabeds from ``uacpy.materials``, same water, same frequency.

    The ladder of critical angles is the whole story of why a hard bottom
    makes a good waveguide and a soft one does not.
    """
    names = ['clay', 'silt', 'sand', 'gravel', 'limestone', 'granite']
    fig, ax = plt.subplots(figsize=WIDE)
    for i, name in enumerate(names):
        bottom = uacpy.BoundaryProperties.from_preset(name)
        rc = reflection(bottom)
        colour = f'C{i}'
        ax.plot(rc.theta, rc.R, color=colour, lw=1.6,
                label=f'{name} ({bottom.sound_speed:.0f} m/s)')
        if bottom.sound_speed > WATER_SPEED:
            theta_c = critical_angle(bottom.sound_speed)
            ax.plot([theta_c], [np.interp(theta_c, rc.theta, rc.R)],
                    'o', color=colour, ms=5.0)
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Grazing angle (°)')
    ax.set_ylabel('|R|')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower center', fontsize=9, ncol=3)
    ax.set_title('Bounce — material presets, 200 Hz (dots mark the critical angle)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig


def shear_loss():
    """The same granite, once as a fluid and once carrying shear.

    Between the shear and compressional critical angles the elastic seabed
    radiates a shear wave the fluid approximation cannot know about, and the
    reflection coefficient drops accordingly.
    """
    fluid = uacpy.BoundaryProperties.from_preset('granite')
    elastic = uacpy.BoundaryProperties.from_preset('granite', elastic=True)

    fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for bottom, label, style in [(fluid, 'fluid (shear dropped)', '--'),
                                 (elastic, 'elastic (c_s = 3000 m/s)', '-')]:
        rc = reflection(bottom)
        ax_r.plot(rc.theta, rc.R, style, lw=1.6, label=label)
        ax_p.plot(rc.theta, np.degrees(rc.phi), style, lw=1.6, label=label)

    theta_s = critical_angle(elastic.shear_speed)
    theta_p = critical_angle(elastic.sound_speed)
    for ax in (ax_r, ax_p):
        ax.axvline(theta_s, color='0.35', lw=1.0,
                   label=f'shear critical {theta_s:.0f}°')
        ax.axvline(theta_p, color='0.35', lw=1.0, ls=':',
                   label=f'compressional critical {theta_p:.0f}°')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 90.0)
    ax_r.set_ylabel('|R|')
    ax_r.legend(loc='lower left', fontsize=9)
    ax_p.set_ylabel('Phase (°)')
    ax_p.set_xlabel('Grazing angle (°)')
    # Both bottoms here are bare half-spaces, and the elastic one cannot be
    # respelled as a layer (BOUNCE marches acoustic layers only), so both phase
    # traces keep the dummy slab's reference. The offset is identical for the
    # two curves, so comparing them is still valid; their absolute level is not.
    ax_p.text(0.012, 0.06,
              'phase referenced 1 m above the seafloor (bare half-space — see Gotchas)',
              transform=ax_p.transAxes, fontsize=8, color='0.35')
    fig.suptitle('Bounce — granite half-space, with and without shear',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def layered_stack():
    """A thin sediment layer turns a smooth mirror into an interference filter."""
    # ``as_explicit_layer`` keeps the bare-granite phase on the same reference
    # plane as the stack's, so the two phase traces are directly comparable.
    bare = as_explicit_layer(uacpy.BoundaryProperties.from_preset('granite'))
    stack = uacpy.SeabedColumn.from_presets(
        layers=[('sand', 8.0)], halfspace='granite')

    fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for bottom, label, style in [(bare, 'bare granite', '--'),
                                 (stack, '8 m sand over granite', '-')]:
        rc = reflection(bottom)
        ax_r.plot(rc.theta, rc.R, style, lw=1.6, label=label)
        ax_p.plot(rc.theta, np.degrees(rc.phi), style, lw=1.6, label=label)
    for ax in (ax_r, ax_p):
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 90.0)
    ax_r.set_ylabel('|R|')
    ax_r.set_ylim(0.0, 1.05)
    ax_r.legend(loc='lower left', fontsize=9)
    ax_p.set_ylabel('Phase (°)')
    ax_p.set_xlabel('Grazing angle (°)')
    fig.suptitle('Bounce — the imprint of one 8 m sediment layer, 200 Hz',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def frequency_dependence():
    """|R(θ, f)| for the layered stack, swept one frequency at a time.

    BOUNCE tabulates one frequency per run and picks its own angular grid
    (denser at high frequency), so the sweep resamples each row onto a common
    angle axis before stacking.
    """
    env, _, _ = layered_elastic()
    freqs = np.linspace(100.0, 2000.0, 72)
    theta = np.linspace(0.0, 90.0, 601)
    R = np.empty((theta.size, freqs.size))
    for j, f in enumerate(freqs):
        source = uacpy.Source(depths=25.0, frequencies=float(f))
        rc = Bounce(**GRID).run(env, source, PROBE)
        R[:, j] = np.interp(theta, rc.theta, rc.R)

    fig, ax = plt.subplots(figsize=TALL)
    im = ax.pcolormesh(freqs, theta, R, shading='nearest', cmap='viridis',
                       vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label='|R|')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Grazing angle (°)')
    ax.set_title('Bounce — 8 m sand over granite, |R(θ, f)|',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig


def bellhop_route():
    """How Bounce reaches a user who never calls it: Bellhop's auto-BOUNCE.

    Same layered seabed twice — once turned into a reflection table by BOUNCE,
    once flattened to an equivalent fluid half-space.
    """
    env, source, receiver = layered_elastic()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True, sharey=True)
    labels = ['auto_bounce=True — BOUNCE reflection table',
              'auto_bounce=False — bottom flattened to a fluid half-space']
    for ax, auto, label in zip(axes, (True, False), labels):
        tl = Bellhop(n_beams=6000, auto_bounce=auto).run(
            env, source, receiver, run_mode=RunMode.COHERENT_TL)
        tl.plot(env=env, ax=ax, show_colorbar=(ax is axes[0]))
        ax.set_title(label, fontweight='bold', fontsize=10)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('Bellhop over a layered seabed, with and without auto-BOUNCE',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


FIGURES = {
    'bounce_reflection': reflection_coefficient,
    'bounce_attenuation': attenuation_ladder,
    'bounce_materials': material_presets,
    'bounce_shear': shear_loss,
    'bounce_layered': layered_stack,
    'bounce_frequency': frequency_dependence,
    'bounce_bellhop': bellhop_route,
}
