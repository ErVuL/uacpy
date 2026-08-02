"""Figures for ``docs/guide/noise.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import uacpy
from uacpy.acoustic_signal.bands import decidecade_bands
from uacpy.noise import (
    HEARING_GROUPS,
    RNL_UNCERTAINTY_DB,
    WenzNoise,
    apply_weighting,
    lloyd_mirror_correction,
    monopole_source_level,
    nominal_source_depth,
    radiated_noise_level,
    weighted_level,
)

GUIDE = True

# Colours matching uacpy.plot.plot_wenz, so the shaded regime bands on the
# composite figure agree with the component lines they sit under.
REGIME_COLOURS = {'turbulence': 'purple', 'shipping': 'blue',
                  'wind': 'green', 'thermal': 'red'}


def _shade_regimes(ax, wenz):
    """Shade the band each physical source dominates, read off the components."""
    names = list(REGIME_COLOURS)
    dominant = np.argmax(np.stack([getattr(wenz, n) for n in names]), axis=0)
    f = wenz.frequencies
    breaks = np.flatnonzero(np.diff(dominant)) + 1
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, f.size]):
        name = names[dominant[lo]]
        ax.axvspan(f[lo], f[hi - 1], color=REGIME_COLOURS[name], alpha=0.07,
                   zorder=0)
        ax.text(np.sqrt(f[lo] * f[hi - 1]), 141.0, name, ha='center', va='top',
                fontsize=9, fontweight='bold', color=REGIME_COLOURS[name])


def wenz_composite():
    """The Wenz composite with every component drawn under the envelope."""
    f = np.logspace(0.0, 5.3, 1200)              # 1 Hz - 200 kHz
    wenz = WenzNoise(f, wind_speed=10.0, shipping_level='medium')

    fig, ax = uacpy.plot.plot_wenz(
        wenz, figsize=(9.6, 5.6),
        title='Wenz ambient noise — 10 kn wind, medium shipping, no rain')
    _shade_regimes(ax, wenz)
    return fig


def wind_sea_state():
    """Wind noise against wind speed: spectra, then level at three frequencies."""
    f = np.logspace(1.0, 5.0, 800)
    speeds = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    fig, (ax_s, ax_u) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cmap = plt.get_cmap('viridis')
    for i, u in enumerate(speeds):
        wenz = WenzNoise(f, wind_speed=u, shipping_level='no')
        ax_s.semilogx(f, wenz.wind, color=cmap(i / (len(speeds) - 1)),
                      label=f'{u:g} kn')
    ax_s.set_xlabel('Frequency [Hz]')
    ax_s.set_ylabel(r'Wind noise [dB re 1 µPa$^2$/Hz]')
    ax_s.set_title('Wind component vs wind speed', loc='left',
                   fontweight='bold', fontsize=11)
    ax_s.grid(which='both', alpha=0.3)
    ax_s.legend(fontsize=8, ncol=2)

    probes = np.array([100.0, 1000.0, 10_000.0])
    u_grid = np.linspace(1.0, 40.0, 160)
    levels = np.array([WenzNoise(probes, wind_speed=u).wind for u in u_grid])
    for j, fp in enumerate(probes):
        label = f'{fp:.0f} Hz' if fp < 1000 else f'{fp / 1000:.0f} kHz'
        ax_u.plot(u_grid, levels[:, j], color=f'C{j}', label=label)
    # Beaufort force from the wind-speed bands in the WenzNoise docstring.
    for force, (lo, hi) in enumerate(
            [(1, 3), (4, 6), (7, 10), (11, 16), (17, 21), (22, 27), (28, 33),
             (34, 40)], start=1):
        ax_u.axvline(hi + 0.5, color='0.8', linewidth=0.7, zorder=0)
        ax_u.text(np.sqrt(lo * hi), 73.0, f'B{force}', ha='center', va='top',
                  fontsize=7.5, color='0.45')
    ax_u.set_xlim(1.0, 40.0)
    ax_u.set_ylim(20.0, 75.0)
    ax_u.set_xlabel('Wind speed [kn]')
    ax_u.set_ylabel(r'Wind noise [dB re 1 µPa$^2$/Hz]')
    ax_u.set_title('Level vs wind speed, with Beaufort force', loc='left',
                   fontweight='bold', fontsize=11)
    ax_u.grid(alpha=0.3)
    ax_u.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    return fig


def shipping_and_rain():
    """The two categorical source strengths: traffic density and rain rate."""
    f = np.logspace(0.0, 4.5, 900)

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for i, level in enumerate(['low', 'medium', 'high']):
        deep = WenzNoise(f, wind_speed=5.0, shipping_level=level)
        shallow = WenzNoise(f, wind_speed=5.0, shipping_level=level,
                            water_depth='shallow')
        ax_s.semilogx(f, deep.shipping, color=f'C{i}', label=f'{level} (deep)')
        ax_s.semilogx(f, shallow.shipping, color=f'C{i}', linestyle='--',
                      linewidth=1.2, label=f'{level} (shallow)')
    ax_s.set_ylim(20.0, 100.0)
    ax_s.set_xlabel('Frequency [Hz]')
    ax_s.set_ylabel(r'Shipping noise [dB re 1 µPa$^2$/Hz]')
    ax_s.set_title('Shipping — traffic density and water depth', loc='left',
                   fontweight='bold', fontsize=11)
    ax_s.grid(which='both', alpha=0.3)
    ax_s.legend(fontsize=8, ncol=2)

    for i, rate in enumerate(['light', 'moderate', 'heavy', 'veryheavy']):
        wenz = WenzNoise(f, wind_speed=5.0, rain_rate=rate)
        ax_r.semilogx(f, wenz.rain, color=f'C{i}', label=rate)
    ax_r.semilogx(f, WenzNoise(f, wind_speed=15.0).wind, color='green',
                  linestyle=':', label='wind at 15 kn')
    ax_r.axvline(7000.0, color='0.5', linewidth=0.9, linestyle='-.')
    ax_r.text(6300.0, 96.0, 'cubic fit valid to 7 kHz;\n−5 dB/octave above',
              fontsize=7.5, color='0.35', ha='right', va='top')
    ax_r.set_ylim(20.0, 100.0)
    ax_r.set_xlabel('Frequency [Hz]')
    ax_r.set_ylabel(r'Rain noise [dB re 1 µPa$^2$/Hz]')
    ax_r.set_title('Rain — rate, against a 15 kn wind', loc='left',
                   fontweight='bold', fontsize=11)
    ax_r.grid(which='both', alpha=0.3)
    ax_r.legend(fontsize=8)

    fig.tight_layout()
    return fig


def component_models():
    """What swapping a registry entry changes: wind and shipping submodels."""
    f = np.logspace(1.0, 5.0, 800)

    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for i, u in enumerate([10.0, 25.0]):
        merk = WenzNoise(f, wind_speed=u, wind_model='merklinger')
        coat = WenzNoise(f, wind_speed=u, wind_model='coates')
        ax_w.semilogx(f, merk.wind, color=f'C{i}',
                      label=f"merklinger, {u:g} kn")
        ax_w.semilogx(f, coat.wind, color=f'C{i}', linestyle='--',
                      label=f"coates, {u:g} kn")
    ax_w.set_ylim(10.0, 80.0)
    ax_w.set_xlabel('Frequency [Hz]')
    ax_w.set_ylabel(r'Wind noise [dB re 1 µPa$^2$/Hz]')
    ax_w.set_title("WIND_MODELS", loc='left', fontweight='bold', fontsize=11)
    ax_w.grid(which='both', alpha=0.3)
    ax_w.legend(fontsize=8)

    for i, level in enumerate(['low', 'high']):
        wenz_ship = WenzNoise(f, wind_speed=10.0, shipping_level=level,
                              shipping_model='wenz')
        coat_ship = WenzNoise(f, wind_speed=10.0, shipping_level=level,
                              shipping_model='coates')
        ax_s.semilogx(f, wenz_ship.shipping, color=f'C{i}',
                      label=f'wenz, {level}')
        ax_s.semilogx(f, coat_ship.shipping, color=f'C{i}', linestyle='--',
                      label=f'coates, {level}')
    ax_s.set_ylim(10.0, 100.0)
    ax_s.set_xlabel('Frequency [Hz]')
    ax_s.set_ylabel(r'Shipping noise [dB re 1 µPa$^2$/Hz]')
    ax_s.set_title('SHIPPING_MODELS', loc='left', fontweight='bold',
                   fontsize=11)
    ax_s.grid(which='both', alpha=0.3)
    ax_s.legend(fontsize=8)

    fig.tight_layout()
    return fig


def ship_source_level():
    """A measured ship spectrum: RNL, its uncertainty, and the monopole MSL."""
    _, fc, _ = decidecade_bands(10.0, 25_000.0)
    received = 130.0 - 18.0 * np.log10(np.maximum(fc / 60.0, 1.0))
    rnl = radiated_noise_level(received, 150.0)
    d_s = nominal_source_depth(8.0)
    msl = monopole_source_level(rnl, fc, d_s)

    sigma = np.where(fc <= 100.0, RNL_UNCERTAINTY_DB['low'],
                     np.where(fc < 20_000.0, RNL_UNCERTAINTY_DB['mid'],
                              RNL_UNCERTAINTY_DB['high']))

    fig, (ax_l, ax_d) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    uacpy.plot.plot_source_level(
        fc, msl, ax=ax_l, label=f'MSL, $d_s$ = {d_s:g} m (ISO 17208-2)',
        title='Ship radiated noise — measured RNL to monopole source level')
    ax_l.semilogx(fc, rnl, color='0.35', linestyle='--', marker='',
                  label='RNL (ISO 17208-1)')
    ax_l.fill_between(fc, rnl - sigma, rnl + sigma, color='0.6', alpha=0.25,
                      label='RNL_UNCERTAINTY_DB')
    ax_l.set_xlabel('')
    ax_l.legend(fontsize=8)

    for d in [2.8, 5.6, 10.5]:
        ax_d.semilogx(fc, lloyd_mirror_correction(fc, d),
                      label=f'$d_s$ = {d:g} m')
    ax_d.axhline(-3.01, color='0.4', linestyle=':', linewidth=1.0)
    ax_d.text(12.0, -2.7, 'incoherent limit, −3.01 dB', fontsize=8,
              color='0.35')
    ax_d.axhline(0.0, color='0.8', linewidth=0.8, zorder=0)
    ax_d.set_ylim(-6.0, 25.0)
    ax_d.set_xlabel('Decidecade band centre [Hz]')
    ax_d.set_ylabel(r'$\Delta L = L_s - L_{RN}$ [dB]')
    ax_d.set_title('Lloyd-mirror correction vs nominal source depth',
                   loc='left')
    ax_d.grid(which='both', alpha=0.3)
    ax_d.legend(fontsize=8)

    fig.tight_layout()
    return fig


def weighting_groups():
    """The eight Southall (2019) auditory weighting functions."""
    in_water = ['LF', 'HF', 'VHF', 'SI', 'PCW', 'OCW']
    in_air = [g for g in HEARING_GROUPS if g not in in_water]

    fig, ax = uacpy.plot.plot_weighting(in_water, figsize=(9.0, 4.8))
    uacpy.plot.plot_weighting(
        in_air, ax=ax, linestyle='--',
        title='Marine-mammal auditory weighting (Southall et al. 2019) — '
              'in-water solid, in-air dashed')
    ax.legend(fontsize=8, ncol=2, loc='lower center')
    return fig


def weighted_soundscape():
    """What weighting changes: the same ambient spectrum, six ways of hearing it."""
    f = np.logspace(1.0, 5.0, 1200)              # 10 Hz - 100 kHz
    wenz = WenzNoise(f, wind_speed=10.0, shipping_level='medium')
    in_water = ['LF', 'HF', 'VHF', 'SI', 'PCW', 'OCW']

    unweighted = 10.0 * np.log10(np.trapezoid(10.0 ** (wenz.total / 10.0), f))
    weighted = {g: weighted_level(wenz.total, f, g) for g in in_water}

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax_s.semilogx(f, wenz.total, color='black', linewidth=2.0,
                  label=f'unweighted ({unweighted:.0f} dB)')
    for i, g in enumerate(['LF', 'HF', 'VHF']):
        ax_s.semilogx(f, apply_weighting(wenz.total, f, g), color=f'C{i}',
                      label=f'{g}-weighted ({weighted[g]:.0f} dB)')
    ax_s.set_ylim(0.0, 100.0)
    ax_s.set_xlabel('Frequency [Hz]')
    ax_s.set_ylabel(r'Level density [dB re 1 µPa$^2$/Hz]')
    ax_s.set_title('Ambient spectrum, weighted', loc='left',
                   fontweight='bold', fontsize=11)
    ax_s.grid(which='both', alpha=0.3)
    ax_s.legend(fontsize=8)

    ax_b.bar(range(len(in_water)), [weighted[g] for g in in_water],
             color=[f'C{i}' for i in range(len(in_water))])
    ax_b.axhline(unweighted, color='black', linewidth=1.6)
    ax_b.text(-0.4, unweighted + 0.5, f'unweighted, {unweighted:.1f} dB',
              fontsize=8)
    for i, g in enumerate(in_water):
        ax_b.text(i, weighted[g] + 0.6, f'{weighted[g] - unweighted:+.1f}',
                  ha='center', fontsize=8)
    ax_b.set_xticks(range(len(in_water)), in_water)
    ax_b.set_ylim(75.0, 102.0)
    ax_b.set_ylabel(r'Broadband level [dB re 1 µPa$^2$]')
    ax_b.set_title('Same ocean, per hearing group', loc='left',
                   fontweight='bold', fontsize=11)
    ax_b.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    return fig


FIGURES = {
    'noise_wenz_composite': wenz_composite,
    'noise_wind_sea_state': wind_sea_state,
    'noise_shipping_rain': shipping_and_rain,
    'noise_component_models': component_models,
    'noise_ship_source_level': ship_source_level,
    'noise_weighting_groups': weighting_groups,
    'noise_weighted_soundscape': weighted_soundscape,
}
