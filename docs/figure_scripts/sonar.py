"""Figures for ``docs/guide/sonar.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

The model-driven budgets run over the shared ``deep_water`` and
``shallow_water`` channels from :mod:`figure_scripts._common`, so the
propagation is the same water the model pages draw. The closed-form scattering
and target-strength laws need a higher-frequency sonar than those channels use
— Chapman & Harris is fitted over 0.4-6.4 kHz and the geometric target-strength
forms need ``ka >> 1`` — so they run at 5 kHz over the same 100 m geometry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, deep_water, shallow_water

import uacpy
from uacpy import sonar
from uacpy.models import Bellhop, Kraken, RunMode
from uacpy.visualization.plots import (
    plot_detection_probability,
    plot_roc,
    plot_signal_excess,
)

GUIDE = True

# ── Detection thresholds, from the (P_D, P_F) each sonar is designed for ─────
DT_PASSIVE = sonar.detection_threshold_energy(
    0.5, 1e-4, bandwidth_hz=50.0, integration_time_s=10.0)
DT_ACTIVE = sonar.detection_threshold_energy(
    0.5, 1e-4, bandwidth_hz=100.0, integration_time_s=0.5)

# ── Two passive budgets. Every level shares one band reference (dB re 1 µPa²
# per Hz here), which is what makes the differences meaningful.
PASSIVE_DEEP = dict(source_level=125.0, noise_level=65.0,
                    directivity_index=15.0, detection_threshold=DT_PASSIVE)
PASSIVE_SHELF = dict(source_level=110.0, noise_level=75.0,
                     directivity_index=15.0, detection_threshold=DT_PASSIVE)

SIGMA_DB = 5.6          # Dyer's saturated-multipath signal-excess fluctuation
CUT_DEPTH = 60.0        # m — the depth the shelf range cut is taken at

# ── The active budget: a 170 dB / 200 Hz sonar over the shelf ────────────────
ACTIVE_SL = 170.0
ACTIVE_NL = 60.0
DIRECTIVITY = 15.0
PULSE_S = 0.5
BEAMWIDTH_RAD = 0.3
SONAR_HEIGHT = 75.0     # m above the seabed — sets the grazing angle vs range

# ── Matched-field geometry ───────────────────────────────────────────────────
ARRAY_DEPTHS = np.linspace(5.0, 95.0, 16)       # 16-element vertical array
TRUE_DEPTH, TRUE_RANGE = 62.0, 3200.0           # the source we have to find
CAND_DEPTHS = np.linspace(5.0, 95.0, 181)
CAND_RANGES = np.linspace(500.0, 5000.0, 226)


# ── shared runs ──────────────────────────────────────────────────────────────


def _passive_deep():
    """Deep-water Bellhop TL turned into a passive signal-excess field."""
    env, source, receiver = deep_water()
    tl = Bellhop(beam_type='G', n_beams=0).run(
        env, source, receiver, run_mode=RunMode.INCOHERENT_TL)
    return env, sonar.passive_signal_excess_field(tl, **PASSIVE_DEEP)


def _passive_shelf():
    """A passive budget over the shallow-water channel, at 200 Hz.

    Kraken rather than Bellhop: 100 m of water at 200 Hz is ``D/λ ≈ 13``, and
    over tens of kilometres the ray fan thins out where the modal sum stays
    smooth. Nothing in :mod:`uacpy.sonar` cares which model produced the TL.
    """
    env, source, receiver = shallow_water()
    tl = Kraken().run(env, source, receiver, run_mode=RunMode.INCOHERENT_TL)
    return env, tl, sonar.passive_signal_excess_field(tl, **PASSIVE_SHELF)


def _csdm(modes, seed=0, snr_db=10.0, n_snapshots=50):
    """A measured CSDM: one source at ``(TRUE_DEPTH, TRUE_RANGE)`` plus noise.

    ``modes`` describes the ocean the *data* came from — pass a perturbed
    environment's modes to simulate environmental mismatch.
    """
    e = sonar.synthesize_replica(modes, TRUE_DEPTH, TRUE_RANGE, ARRAY_DEPTHS)
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n_snapshots))
    n_power = np.mean(np.abs(e) ** 2) / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(n_power / 2.0) * (
        rng.standard_normal((ARRAY_DEPTHS.size, n_snapshots))
        + 1j * rng.standard_normal((ARRAY_DEPTHS.size, n_snapshots)))
    return sonar.csdm(e[:, None] * phases + noise)


def _draw_surface(ax, surface, title, *, extra='', floor_db=-20.0):
    """One normalised ambiguity surface with truth and estimate marked."""
    db = 10.0 * np.log10(np.clip(surface / np.nanmax(surface),
                                 10.0 ** (floor_db / 10.0), None))
    im = ax.imshow(
        db, aspect='auto', cmap='turbo', vmin=floor_db, vmax=0.0,
        extent=[CAND_RANGES[0] / 1e3, CAND_RANGES[-1] / 1e3,
                CAND_DEPTHS[-1], CAND_DEPTHS[0]])
    iz, ir = np.unravel_index(np.nanargmax(surface), surface.shape)
    ax.plot(TRUE_RANGE / 1e3, TRUE_DEPTH, '*', mfc='w', mec='k', ms=15,
            mew=0.8, label='true source')
    ax.plot(CAND_RANGES[ir] / 1e3, CAND_DEPTHS[iz], 'o', mfc='none', mec='k',
            ms=15, mew=3.0, label='estimate')
    ax.plot(CAND_RANGES[ir] / 1e3, CAND_DEPTHS[iz], 'o', mfc='none', mec='w',
            ms=15, mew=1.2)
    ax.set_title(f'{title}\nestimate ({CAND_DEPTHS[iz]:.0f} m, '
                 f'{CAND_RANGES[ir] / 1e3:.2f} km){extra}', fontsize=10)
    return im


# ── figures ──────────────────────────────────────────────────────────────────


def signal_excess_field():
    """The hero: where a passive sonar can and cannot detect.

    Signal excess is the sonar equation evaluated at every cell of a modelled
    TL grid. Warm is detectable, cool is not, and the black contour is the
    SE = 0 detection boundary.
    """
    env, se = _passive_deep()
    fig, _ = plot_signal_excess(
        se, env=env, figsize=WIDE,
        title='Passive signal excess over a Bellhop TL field — '
              'deep water, 50 Hz')
    return fig


def detection_range():
    """The figure of merit as a TL budget, and the range profile it implies."""
    _, tl, se = _passive_shelf()
    fom = sonar.figure_of_merit(**PASSIVE_SHELF)
    cut = se.at(depth=CUT_DEPTH)
    r_det = sonar.detection_range(cut.coords['range'], cut.data)
    depths, ranges = sonar.detection_range_by_depth(se)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    tl_cut = tl.at(depth=CUT_DEPTH)
    ax.plot(tl_cut.coords['range'] / 1e3, tl_cut.data, color='C0', lw=1.2,
            label='one-way TL')
    ax.axhline(fom, color='C3', ls='--', lw=1.5, label=f'FOM = {fom:.1f} dB')
    ax.axvline(r_det / 1e3, color='k', ls=':', lw=1.2,
               label=f'detection range = {r_det / 1e3:.2f} km')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Transmission loss (dB)')
    ax.set_title(f'TL against the figure of merit, {CUT_DEPTH:.0f} m',
                 fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    ax.plot(ranges / 1e3, depths, color='C0', lw=1.4)
    ax.set_xlabel('Detection range (km)')
    ax.set_ylabel('Receiver depth (m)')
    ax.set_title('Detection range vs depth (SE = 0)', fontsize=10)
    ax.set_xlim(0.0, 5.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure of merit and detection range — shelf channel, 200 Hz',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def detection_probability_field():
    """P_D from signal excess, for two channel-fluctuation models."""
    env, se = _passive_deep()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    for ax, sigma in zip(axes, (SIGMA_DB, 9.0)):
        pd = sonar.probability_of_detection_field(se, sigma_db=sigma)
        plot_detection_probability(
            pd, ax=ax, env=env,
            title=f'Detection probability, σ = {sigma:g} dB')
    axes[0].set_xlabel('')
    fig.suptitle('The SE = 0 boundary is a transition band, not an edge',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def detection_theory():
    """From a required (P_D, P_F) to the DT term of the sonar equation."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    plot_roc([1.0, 2.0, 3.0, 4.0, 5.0], ax=axes[0],
             title='ROC — Gaussian detector')
    axes[0].axvline(1e-4, color='k', ls='--', lw=1.0, alpha=0.6)
    axes[0].text(1.3e-4, 0.06, '$P_F = 10^{-4}$', fontsize=8)

    ax = axes[1]
    pd_axis = np.linspace(0.1, 0.99, 200)
    for pf, colour in ((1e-2, 'C0'), (1e-4, 'C1'), (1e-6, 'C3')):
        d = [sonar.detection_index(p, pf) for p in pd_axis]
        ax.plot(pd_axis, 10.0 * np.log10(d), color=colour,
                label=f'$P_F = 10^{{{int(np.log10(pf))}}}$')
    ax.set_xlabel('Required $P_D$')
    ax.set_ylabel('$10\\log_{10} d$  (dB)')
    ax.set_title('Detection index the detector must reach', loc='left',
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[2]
    wt = np.logspace(0.0, 5.0, 200)
    for pd_target, colour in ((0.5, 'C0'), (0.9, 'C3')):
        dt = [sonar.detection_threshold_energy(
                  pd_target, 1e-4, bandwidth_hz=m, integration_time_s=1.0)
              for m in wt]
        ax.semilogx(wt, dt, color=colour, label=f'$P_D$ = {pd_target}')
    ax.set_xlabel('Time-bandwidth product  $wt$')
    ax.set_ylabel('Detection threshold  DT (dB)')
    ax.set_title('DT falls 5 dB per decade of integration', loc='left',
                 fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Detection theory — where DT comes from', fontweight='bold',
                 fontsize=12)
    fig.tight_layout()
    return fig


def target_strength():
    """Which shapes care about aspect, and which do not."""
    freq = 5000.0
    angles = np.linspace(-30.0, 30.0, 2401)

    plate = sonar.ts_plate(4.0, 2.0, freq, angle_deg=angles)
    cylinder = sonar.ts_cylinder(1.0, 8.0, freq, angle_deg=angles)
    sphere = sonar.ts_sphere(2.0, frequency_hz=freq)
    ellipsoid = sonar.ts_ellipsoid(20.0, 4.0, 4.0, frequency_hz=freq)

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(angles, plate, color='C0', lw=0.9,
            label='plate 4 × 2 m — angle from normal incidence')
    ax.plot(angles, cylinder, color='C3', lw=0.9,
            label='cylinder a = 1 m, L = 8 m — angle from broadside')
    ax.axhline(sphere, color='C2', ls='--', lw=1.6,
               label=f'sphere a = 2 m — {sphere:.1f} dB at every aspect')
    ax.axhline(ellipsoid, color='C1', ls=':', lw=1.8,
               label=f'ellipsoid 20 × 4 × 4 m end-on — {ellipsoid:.1f} dB')
    ax.set_xlabel('Aspect angle (degrees)')
    ax.set_ylabel('Target strength (dB re 1 m²)')
    ax.set_title('Geometric-regime target strength at 5 kHz',
                 fontweight='bold', fontsize=12)
    ax.set_ylim(-20.0, 40.0)
    ax.set_xlim(angles[0], angles[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig


def reverberation():
    """Boundary and volume reverberation decaying after a 5 kHz ping.

    The components fall off at different rates because their scattering cells
    grow differently — the boundary cell as ``r``, the volume cell as ``r²`` —
    and because the scattering strength itself follows the grazing angle down.
    """
    ranges = np.linspace(50.0, 8000.0, 700)
    sound_speed = 1500.0
    two_way_time = 2.0 * ranges / sound_speed
    grazing = np.rad2deg(np.arctan2(50.0, ranges))   # sonar 50 m off each boundary

    bottom = sonar.boundary_reverberation(
        ranges, 210.0, sonar.lambert_bottom(grazing),
        pulse_length_s=0.02, horizontal_beamwidth_rad=0.15,
        sound_speed=sound_speed)
    surface = sonar.boundary_reverberation(
        ranges, 210.0,
        sonar.chapman_harris_surface(grazing, wind_speed_kn=25.0,
                                     frequency=5000.0),
        pulse_length_s=0.02, horizontal_beamwidth_rad=0.15,
        sound_speed=sound_speed)
    volume = sonar.volume_reverberation(
        ranges, 210.0, -70.0,
        pulse_length_s=0.02, solid_angle_beamwidth_sr=0.01,
        sound_speed=sound_speed)
    total = sonar.total_reverberation(bottom, surface, volume)
    background = sonar.noise_background(55.0, 20.0)

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(two_way_time, total, color='0.7', lw=5.0, solid_capstyle='round',
            zorder=1, label='total (incoherent sum)')
    ax.plot(two_way_time, surface, color='C0', lw=1.4, zorder=2,
            label='surface — Chapman–Harris, 25 kn')
    ax.plot(two_way_time, bottom, color='C1', lw=1.4, zorder=2,
            label='bottom — Lambert, µ = −27 dB')
    ax.plot(two_way_time, volume, color='C2', lw=1.4, zorder=2,
            label='volume — $S_v$ = −70 dB re 1/m')
    ax.axhline(background, color='k', ls='--', lw=1.2, zorder=2,
               label='noise background  NL − DI')
    ax.set_xlabel('Two-way travel time (s)')
    ax.set_ylabel('Reverberation level (dB)')
    ax.set_title('Reverberation decay after a 20 ms ping at 5 kHz',
                 fontweight='bold', fontsize=12)
    ax.set_ylim(background - 20.0, None)
    ax.set_xlim(0.0, two_way_time[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    secondary = ax.secondary_xaxis(
        'top', functions=(lambda t: t * sound_speed / 2e3,
                          lambda r: 2e3 * r / sound_speed))
    secondary.set_xlabel('Range (km)')
    fig.tight_layout()
    return fig


def active_budget():
    """The active sonar equation, term by term, over a modelled TL field.

    Echo level against the two competing backgrounds: ambient noise, flat with
    range, and bottom reverberation, which is a scaled copy of the echo. Inside
    the crossover the sonar is reverberation-limited, and raising the source
    level buys nothing at all.
    """
    env, source, _ = shallow_water()
    receiver = uacpy.Receiver(depths=np.linspace(1.0, 99.0, 60),
                              ranges=np.linspace(200.0, 30_000.0, 300))
    tl_field = Kraken().run(env, source, receiver,
                            run_mode=RunMode.INCOHERENT_TL)
    ranges = receiver.ranges
    tl = tl_field.at(depth=CUT_DEPTH).data
    tl_bottom = tl_field.at(depth=99.0).data

    ts = sonar.ts_cylinder(2.0, 8.0, 200.0)
    el = sonar.echo_level(ACTIVE_SL, tl, ts)
    nl_di = sonar.noise_background(ACTIVE_NL, DIRECTIVITY)

    # The scattering patch is on the seabed, so its grazing angle falls as the
    # range grows — that is what makes reverberation a near-field problem.
    grazing = np.rad2deg(np.arctan2(SONAR_HEIGHT, ranges))
    rl = sonar.boundary_reverberation(
        ranges, ACTIVE_SL, sonar.lambert_bottom(grazing),
        pulse_length_s=PULSE_S, horizontal_beamwidth_rad=BEAMWIDTH_RAD,
        tl_db=tl_bottom)

    se_noise = sonar.active_signal_excess(
        ACTIVE_SL, tl, ts, noise_level=ACTIVE_NL,
        directivity_index=DIRECTIVITY, detection_threshold=DT_ACTIVE)
    se_both = sonar.active_signal_excess(
        ACTIVE_SL, tl, ts, noise_level=ACTIVE_NL,
        directivity_index=DIRECTIVITY, reverberation_level=rl,
        detection_threshold=DT_ACTIVE)
    crossover = float(ranges[np.argmax(rl < nl_di)])

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for ax in axes:
        ax.axvspan(ranges[0] / 1e3, crossover / 1e3, color='C3', alpha=0.08)
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)

    ax = axes[0]
    ax.plot(ranges / 1e3, el, color='C0', lw=1.4,
            label='echo level  EL = SL − 2·TL + TS')
    ax.plot(ranges / 1e3, rl, color='C3', lw=1.4,
            label='reverberation  RL')
    ax.axhline(nl_di, color='C2', ls='--', lw=1.4, label='noise  NL − DI')
    ax.set_ylabel('Level (dB)')
    ax.set_title(f'Echo against its two backgrounds — TS = {ts:.1f} dB, '
                 f'reverberation-limited inside {crossover / 1e3:.1f} km',
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    for se, colour, label in ((se_noise, 'C2', 'noise only'),
                              (se_both, 'C3', 'noise + reverberation')):
        r_det = sonar.detection_range(ranges, se)
        ax.plot(ranges / 1e3, se, color=colour, lw=1.4,
                label=f'{label} — detection range {r_det / 1e3:.1f} km')
        if np.isfinite(r_det):
            ax.axvline(r_det / 1e3, color=colour, ls=':', lw=1.0)
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Signal excess (dB)')
    ax.set_title(f'Reverberation costs {se_noise[0] - se_both[0]:.0f} dB of '
                 f'signal excess at 200 m, and nothing at the detection range',
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle('Active sonar equation over the shelf channel, 200 Hz',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def matched_field():
    """Bartlett vs MVDR, matched and mismatched.

    The data come from a channel that is really 102 m deep; the replicas are
    built for the 100 m on the chart. Bartlett survives with a biased peak;
    MVDR keeps its peak but loses the deep suppression that made it worth
    using.
    """
    env, source, _ = shallow_water()
    modes = Kraken().compute_modes(env, source)
    bank = sonar.replica_bank(modes, ARRAY_DEPTHS, CAND_DEPTHS, CAND_RANGES)

    true_env = uacpy.Environment(
        name='The ocean, 2 m deeper than charted',
        bathymetry=102.0,
        ssp=[(0.0, 1500.0), (30.0, 1495.0), (102.0, 1490.0)],
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1650.0, density=1.8, attenuation=0.6,
        ),
    )
    k_matched = _csdm(modes)
    k_mismatched = _csdm(Kraken().compute_modes(true_env, source))

    panels = [
        ('Bartlett — matched', sonar.bartlett(k_matched, bank)),
        ('MVDR — matched', sonar.mvdr(k_matched, bank, loading=1e-3)),
        ('Bartlett — 2 m depth mismatch', sonar.bartlett(k_mismatched, bank)),
        ('MVDR — 2 m depth mismatch',
         sonar.mvdr(k_mismatched, bank, loading=1e-3)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True,
                             sharey=True)
    for ax, (title, surface) in zip(axes.ravel(), panels):
        im = _draw_surface(ax, surface, title)
        fig.colorbar(im, ax=ax, label='Normalised power (dB)',
                     fraction=0.046, pad=0.02)
    for ax in axes[1]:
        ax.set_xlabel('Range (km)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Depth (m)')
    axes[0, 0].legend(fontsize=8, loc='upper right', framealpha=0.9)
    fig.suptitle('Matched-field ambiguity surfaces — 16-element VLA, 200 Hz',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def replica_banks():
    """The same localisation with replicas from Kraken modes and from Bellhop.

    A replica is the ocean's Green's function sampled at the array. Modes, rays
    and a PE march are different ways of evaluating it, so any coherent model
    run can fill a replica bank.
    """
    env, source, _ = shallow_water()
    modes = Kraken().compute_modes(env, source)
    modal_bank = sonar.replica_bank(modes, ARRAY_DEPTHS, CAND_DEPTHS,
                                    CAND_RANGES)

    # One Bellhop run fills the whole bank: the candidate depths are the source
    # depths, the array elements are the receivers, and the candidate ranges
    # are the range axis. The run returns a ResultStack over source_depth.
    swept = uacpy.Source(depths=CAND_DEPTHS, frequencies=200.0)
    array = uacpy.Receiver(depths=ARRAY_DEPTHS, ranges=CAND_RANGES)
    stack = Bellhop(n_beams=4000).run(env, swept, array,
                                      run_mode=RunMode.COHERENT_TL)
    ray_bank = sonar.replica_bank_from_field(stack, array_depths=ARRAY_DEPTHS)

    K = _csdm(modes)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, title, bank in (
            (axes[0], 'replica_bank — Kraken modes', modal_bank),
            (axes[1], 'replica_bank_from_field — Bellhop', ray_bank)):
        surface = sonar.bartlett(K, bank)
        im = _draw_surface(ax, surface, title,
                           extra=f' · peak {surface.max():.2f}')
        fig.colorbar(im, ax=ax, label='Normalised power (dB)',
                     fraction=0.046, pad=0.02)
        ax.set_xlabel('Range (km)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].legend(fontsize=8, loc='upper right', framealpha=0.9)
    fig.suptitle('Bartlett on the same data, from two replica engines',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


FIGURES = {
    'sonar_signal_excess': signal_excess_field,
    'sonar_detection_range': detection_range,
    'sonar_detection_probability': detection_probability_field,
    'sonar_detection_theory': detection_theory,
    'sonar_target_strength': target_strength,
    'sonar_reverberation': reverberation,
    'sonar_active_budget': active_budget,
    'sonar_matched_field': matched_field,
    'sonar_replica_banks': replica_banks,
}
