"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 38: Matched-Field Source Localization with KRAKEN Replicas
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Localize an unknown source in range and depth by matched-field processing
    (MFP), using replica vectors synthesized directly from a KRAKEN normal-mode
    set — no separate replica engine required.

THEORY:
    A *replica* is the modeled complex pressure at the array sensors for a
    hypothesized source position. The eigenpairs (k_m, phi_m) depend only on the
    environment, so the modes are computed ONCE and every replica in the search
    grid is a cheap analytic re-sum of the KRAKEN far-field modal series:

        p(r, z) = r^-1/2 · Σ_m phi_m(z_s) phi_m(z) k_m^-1/2 exp(-i k_m r)

    MFP scans candidate (depth, range) cells, correlating each replica against
    the measured cross-spectral density matrix (CSDM). The peak is the estimate.

SCENARIO:
    Isovelocity 100 m Pekeris waveguide, 150 Hz, a 16-element vertical array.
    A source at (62 m, 3.2 km) is observed through 50 noisy snapshots (10 dB
    SNR). We localize it with the Bartlett (linear) and MVDR (Capon) processors.

FEATURES DEMONSTRATED:
    ✓ KRAKEN mode computation (compute_modes)
    ✓ Replica synthesis from modes (synthesize_replica / replica_bank)
    ✓ CSDM from array snapshots (csdm)
    ✓ Bartlett vs MVDR ambiguity surfaces (bartlett / mvdr)
    ✓ Range-depth localization peak-picking
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
from pathlib import Path

# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy import Bottom  # noqa: E402
from uacpy.models import Kraken  # noqa: E402
from uacpy.sonar import (  # noqa: E402
    synthesize_replica, replica_bank, csdm, bartlett, mvdr,
)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 38: Matched-field localization with KRAKEN replicas")
    print("═" * 80)

    # ── Environment + array ────────────────────────────────────────────────
    env = uacpy.Environment(
        name="Pekeris 100 m",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(np.array([[0, 1500.0], [100, 1500.0]])),
        bottom=Bottom.from_halfspaces(
            np.array([0.0]), sound_speed=np.array([1800.0]),
            density=np.array([1.8]), attenuation=np.array([0.2]),
            acoustic_type='half-space',
        ),
    )
    source = uacpy.Source(depths=25.0, frequencies=150.0)  # depth here is unused by MFP
    array_depths = np.linspace(5, 95, 16)                  # 16-element VLA

    print("\n[1/4] Computing modes (once)...")
    modes = Kraken(verbose=False).compute_modes(env, source)
    print(f"  ✓ {modes.n_modes} modes at {float(np.atleast_1d(source.frequencies)[0]):.0f} Hz")

    # ── Synthesize noisy array data for a hidden source ────────────────────
    true_z, true_r = 62.0, 3200.0
    print(f"\n[2/4] Simulating data for a hidden source at ({true_z:.0f} m, {true_r/1e3:.1f} km)...")
    e_true = synthesize_replica(modes, true_z, true_r, array_depths)
    rng = np.random.default_rng(1)
    n_snap = 50
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, n_snap))
    sig = e_true[:, None] * phases
    snr_db = 10.0
    npow = np.mean(np.abs(e_true) ** 2) / 10 ** (snr_db / 10)
    noise = np.sqrt(npow / 2) * (
        rng.standard_normal((array_depths.size, n_snap))
        + 1j * rng.standard_normal((array_depths.size, n_snap))
    )
    K = csdm(sig + noise)
    print(f"  ✓ CSDM from {n_snap} snapshots at {snr_db:.0f} dB SNR")

    # ── Replica bank over the search grid + ambiguity surfaces ─────────────
    print("\n[3/4] Building replica bank and scanning...")
    cand_z = np.linspace(5, 95, 91)
    cand_r = np.linspace(500, 5000, 121)
    bank = replica_bank(modes, array_depths, cand_z, cand_r)
    surf_b = bartlett(K, bank)
    # Diagonal loading as a fraction of the average eigenvalue (the mvdr
    # default, stated here because it sets the trade-off): smaller sharpens the
    # Capon peak but makes it brittle to environmental mismatch, larger relaxes
    # the surface back toward Bartlett.
    surf_m = mvdr(K, bank, diagonal_loading=1e-2)

    for name, surf in [("Bartlett", surf_b), ("MVDR", surf_m)]:
        iz, ir = np.unravel_index(np.argmax(surf), surf.shape)
        print(f"  ✓ {name:8s} estimate: depth {cand_z[iz]:5.1f} m, "
              f"range {cand_r[ir]/1e3:.2f} km")

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\n[4/4] Plotting ambiguity surfaces...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    extent = [cand_r[0] / 1e3, cand_r[-1] / 1e3, cand_z[-1], cand_z[0]]
    for ax, name, surf in [(axes[0], "Bartlett", surf_b), (axes[1], "MVDR", surf_m)]:
        db = 10 * np.log10(np.clip(surf / surf.max(), 1e-3, None))
        im = ax.imshow(db, aspect='auto', extent=extent, cmap='turbo',
                       vmin=-15, vmax=0)
        ax.plot(true_r / 1e3, true_z, 'w*', ms=16, mec='k', label='truth')
        iz, ir = np.unravel_index(np.argmax(surf), surf.shape)
        ax.plot(cand_r[ir] / 1e3, cand_z[iz], 'o', mfc='none', mec='w', ms=12,
                mew=2, label='estimate')
        ax.set_title(f"{name} ambiguity surface", loc='left')
        ax.set_xlabel("Range [km]")
        ax.legend(loc='upper right', fontsize=8)
        fig.colorbar(im, ax=ax, label="Normalized power [dB]")
    axes[0].set_ylabel("Depth [m]")
    fig.suptitle("Matched-field localization — KRAKEN replicas (150 Hz, 16-el VLA)")
    out = OUTPUT_DIR / 'example_38_matched_field.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("\n✓ Example 38 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
