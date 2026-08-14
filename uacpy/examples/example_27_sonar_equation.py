"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 27: Sonar Equation, Reverberation & Detection Range
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Turn a transmission-loss curve into sonar performance using ``uacpy.sonar``:
    passive/active signal excess, the noise-vs-reverberation crossover, and the
    detection range where the signal excess reaches zero.

FEATURES DEMONSTRATED:
    ✓ uacpy.sonar.passive_signal_excess / active_signal_excess
    ✓ Lambert bottom backscatter + boundary_reverberation
    ✓ detection_range from a signal-excess curve
    ✓ detection_threshold_energy from (Pd, Pf)
    ✓ passive/active_signal_excess_field on a Bellhop TL grid +
      plot_signal_excess (SE heatmap with the SE = 0 detection boundary)
    ✓ probability_of_detection_field (Urick transition curve, σ = 5.6 dB)
      + plot_detection_probability; detection_range_by_depth profile
    ✓ ts_cylinder — geometric-regime target strength (Urick Table 9.1)
      feeding the active budget

NOTE:
    Part 1 uses a spherical-spreading + Thorp-absorption TL so the sonar-
    equation mechanics are visible with no model binary. Part 2 computes the
    TL field with Bellhop and maps signal excess over the full
    (depth, range) grid.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy import sonar  # noqa: E402
from uacpy.models import Bellhop, RunMode  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_detection_probability,
    plot_signal_excess,
    plot_roc,
)


def thorp_db_per_km(freq_hz):
    """Thorp volume absorption, dB/km, with f in Hz.

    The two small terms differ from ``uacpy.core.absorption.thorp_db_per_km``,
    which carries the JKPS Eq. 1.34 pair (3.0e-4, 3.3e-3) rather than
    (2.75e-4, 0.003). The gap is 4e-4 dB/km at the 2 kHz used here, i.e. under
    0.03 dB over the whole 60 km sweep.
    """
    f = freq_hz / 1000.0
    return (0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2)
            + 2.75e-4 * f**2 + 0.003)


def main():
    print("═" * 80)
    print("EXAMPLE 27: Sonar Equation, Reverberation & Detection Range")
    print("═" * 80)

    freq = 2000.0
    # The range vector has to extend past the SE = 0 crossing or
    # ``detection_range`` has nothing to bracket and returns inf. With
    # SL=140, NL-DI=45 and DT=-4.3 the passive curve crosses near 45 km, so
    # 30 km would stop short of the very quantity this example is about.
    ranges = np.linspace(100.0, 60000.0, 600)
    tl = 20.0 * np.log10(ranges) + thorp_db_per_km(freq) * ranges / 1000.0

    # Detection threshold for Pd=0.5, Pf=1e-4 over a 100 Hz / 1 s integration.
    dt = sonar.detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                          integration_time_s=1.0)

    # --- Passive: detect a 140 dB target against ambient noise ---
    passive_se = sonar.passive_signal_excess(
        source_level=140.0, tl=tl, noise_level=60.0,
        directivity_index=15.0, detection_threshold=dt,
    )
    passive_range = sonar.detection_range(ranges, passive_se)

    # --- Active: echo from a TS=10 dB target, noise vs bottom reverberation ---
    grazing = np.rad2deg(np.arctan2(100.0, ranges))  # 100 m sonar/seafloor height
    sb = sonar.lambert_bottom(grazing)
    rl = sonar.boundary_reverberation(
        ranges, 220.0, sb,
        pulse_length_s=0.05, horizontal_beamwidth_rad=0.1, tl_db=tl,
    )
    active_se = sonar.active_signal_excess(
        220.0, tl, target_strength=10.0, noise_level=60.0,
        directivity_index=15.0, reverberation_level=rl, detection_threshold=dt,
    )
    active_range = sonar.detection_range(ranges, active_se)

    print(f"\n  Detection threshold DT = {dt:.1f} dB")
    print(f"  Passive detection range = {passive_range/1000:.2f} km")
    print(f"  Active  detection range = {active_range/1000:.2f} km")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(ranges / 1000, tl, 'b-')
    ax.set_title('Transmission Loss (spherical + Thorp)', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('TL (dB)')
    ax.invert_yaxis(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ranges / 1000, passive_se, 'g-', label='Passive SE')
    ax.axhline(0, color='k', lw=0.8)
    if np.isfinite(passive_range):
        ax.axvline(passive_range / 1000, color='r', ls='--',
                   label=f'Detection range {passive_range/1000:.1f} km')
    else:
        ax.axhline(np.nan, color='r', ls='--',
                   label='SE never reaches 0 in this range span')
    ax.set_title('Passive Signal Excess', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('SE (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ranges / 1000, rl, 'm-', label='Reverberation level')
    ax.axhline(60.0 - 15.0, color='c', ls='--', label='Noise background (NL-DI)')
    ax.set_title('Active Background: Reverb vs Noise', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('Level (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ranges / 1000, active_se, 'r-', label='Active SE')
    ax.axhline(0, color='k', lw=0.8)
    if np.isfinite(active_range):
        ax.axvline(active_range / 1000, color='b', ls='--',
                   label=f'Detection range {active_range/1000:.1f} km')
    else:
        ax.axhline(np.nan, color='b', ls='--',
                   label='SE never reaches 0 in this range span')
    ax.set_title('Active Signal Excess', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('SE (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'example_27_sonar_equation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

    # ── Part 2: signal-excess map over a model TL grid ───────────────────
    # The TL comes from Bellhop on a 200 m waveguide at 2 kHz. Ray theory is
    # valid here: D/λ ≈ 267, comfortably above the D/λ ≳ 100 ray-regime rule
    # of thumb (modes take over below D/λ ≈ 30 — Stergiopoulos, Advanced
    # Signal Processing Handbook, §10.2.2; docs/models/bellhop.md brackets the
    # same transition more loosely, at 5-20). A
    # mild surface duct (sound-speed maximum near 30 m) traps the near-surface
    # source energy and refracts the rest downward, so the SE map shows real
    # propagation structure — a low-loss surface channel over a weaker
    # sub-duct shadow — rather than plain spreading. (In a flat isovelocity
    # column the incoherent field is genuinely near depth-uniform — correct,
    # but a dull map; a refracting profile is what makes the paths visible.)
    # Incoherent TL is the standard basis for sonar-performance maps: it keeps
    # this ducting/shadow structure (a geometric, ray-density effect) while
    # dropping the fine multipath interference fringes a coherent field would
    # imprint. Budget: a 125 dB target against 75 dB/Hz shipping noise
    # (passive), and a 190 dB / TS = 10 dB active sonar.
    print("\n  Part 2: Bellhop TL grid → signal-excess maps")

    env = uacpy.Environment(
        name="SE grid demo",
        bathymetry=200.0,
        # mild surface duct: sound-speed maximum near 30 m, refracting below
        ssp=[(0.0, 1500.0), (30.0, 1512.0), (120.0, 1496.0), (200.0, 1500.0)],
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, density=1.5, attenuation=0.5,
        ),
    )
    src = uacpy.Source(depths=18.0, frequencies=freq)   # inside the duct
    rcv = uacpy.Receiver(
        depths=np.linspace(0.0, 200.0, 201),
        ranges=np.linspace(100.0, 20000.0, 350),
    )
    # geometric-hat beams with auto beam count / step (NBeams=0), the BELLHOP
    # User Guide's recommendation for TL runs.
    tl_field = Bellhop(beam_type='G', n_beams=0, alpha=(-80, 80)).run(
        env, src, rcv, run_mode=RunMode.INCOHERENT_TL,
    )

    se_passive = sonar.passive_signal_excess_field(
        tl_field, source_level=125.0, noise_level=75.0,
        directivity_index=15.0, detection_threshold=dt,
    )
    # Reverberation uses the same modeled TL as the echo — evaluated at the
    # seafloor depth, where the scattering patch sits.
    tl_at_bottom = tl_field.at(depth=float(env.depth)).db
    rl_grid = sonar.boundary_reverberation(
        rcv.ranges, 190.0,
        sonar.lambert_bottom(np.rad2deg(np.arctan2(100.0, rcv.ranges))),
        pulse_length_s=0.05, horizontal_beamwidth_rad=0.1,
        tl_db=tl_at_bottom,
    )
    # Geometric-regime TS of a 1.5 m × 5 m rigid cylinder at broadside
    # (Urick Table 9.1) instead of an assumed number.
    ts = sonar.ts_cylinder(1.5, 5.0, freq)
    print(f"  Target strength (1.5 m × 5 m cylinder, broadside) = {ts:.1f} dB")
    se_active = sonar.active_signal_excess_field(
        tl_field, source_level=190.0, target_strength=ts,
        noise_level=75.0, directivity_index=15.0,
        reverberation_level=rl_grid, detection_threshold=dt,
    )

    cut = se_passive.at(depth=100.0)
    grid_range = sonar.detection_range(cut.coords['range'], cut.data)
    print(f"  Passive detection range at 100 m depth (Bellhop TL) = "
          f"{grid_range/1000:.2f} km")

    # Mean SE → detection probability via Urick's transition curve
    # (σ = 5.6 dB, Dyer's saturated-multipath fluctuation), and the
    # 50%-detection range profile across the water column.
    pd_passive = sonar.probability_of_detection_field(se_passive, sigma_db=5.6)
    dr_depths, dr_m = sonar.detection_range_by_depth(se_passive)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    plot_signal_excess(
        se_passive, ax=axes[0, 0], env=env,
        title='Passive signal excess',
    )
    plot_signal_excess(
        se_active, ax=axes[0, 1], env=env,
        title='Active signal excess (noise + bottom reverb)',
    )
    plot_detection_probability(
        pd_passive, ax=axes[1, 0], env=env,
        title='Passive detection probability (σ = 5.6 dB)',
    )
    ax = axes[1, 1]
    finite = np.isfinite(dr_m)
    ax.plot(dr_m[finite] / 1000.0, dr_depths[finite], 'b-')
    ax.set_xlabel('Detection range (km)')
    ax.set_ylabel('Receiver depth (m)')
    ax.set_xlim(0, rcv.ranges.max() / 1000.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_title('Passive detection range vs depth (SE = 0)')

    fig.suptitle('Sonar performance over a Bellhop TL grid',
                 fontweight='bold')
    plt.tight_layout()
    out2 = OUTPUT_DIR / 'example_27_signal_excess_grid.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out2}")

    # Receiver operating characteristic (Pd vs Pfa) for a family of detector
    # deflections d' (detection index d = d'^2), via uacpy.visualization.plot_roc
    # (which consumes uacpy.sonar.roc_curve). The dashed line marks the
    # Pfa = 1e-4 operating point used in the panels above.
    fig, ax = plot_roc([1.0, 2.0, 3.0, 4.0, 5.0],
                       title='ROC — Gaussian detector')
    ax.axvline(1e-4, color='k', ls='--', lw=1, alpha=0.6)
    ax.text(1.1e-4, 0.05, 'Pfa = 1e-4', fontsize=8)
    out3 = OUTPUT_DIR / 'example_27_roc.png'
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out3}")
    print("\n✓ Example 27 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
