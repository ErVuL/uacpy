"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 29: Adaptive & High-Resolution Array Processing
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Resolve two closely spaced plane-wave arrivals on a 16-element line array,
    comparing the conventional Bartlett beamformer with the adaptive MVDR
    (Capon) and subspace MUSIC estimators.

FEATURES DEMONSTRATED:
    ✓ uacpy.acoustic_signal.steering_vectors / sample_covariance
    ✓ bartlett_spectrum / mvdr_spectrum / music_spectrum
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.acoustic_signal import (  # noqa: E402
    bartlett_spectrum,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    steering_vectors,
)


def main():
    print("═" * 80)
    print("EXAMPLE 29: Adaptive & High-Resolution Array Processing")
    print("═" * 80)

    freq, c = 1500.0, 1500.0
    n_elem = 16
    positions = np.arange(n_elem) * (c / freq / 2.0)  # half-wavelength spacing
    true_angles = [-8.0, 6.0]

    rng = np.random.default_rng(0)
    n_snap = 500
    x = np.zeros((n_elem, n_snap), dtype=complex)
    for ang in true_angles:
        a = steering_vectors(positions, [ang], freq, c)[0]
        src = rng.standard_normal(n_snap) + 1j * rng.standard_normal(n_snap)
        x += np.outer(a, src)
    x += 0.05 * (rng.standard_normal((n_elem, n_snap))
                 + 1j * rng.standard_normal((n_elem, n_snap)))

    R = sample_covariance(x)
    angles = np.linspace(-40, 40, 801)
    bart = bartlett_spectrum(R, positions, angles, freq, c)
    capon = mvdr_spectrum(R, positions, angles, freq, c)
    music = music_spectrum(R, positions, angles, freq, n_sources=2, c=c)

    def norm_db(p):
        return 10 * np.log10(p / p.max())

    print(f"\n  True arrival angles: {true_angles} deg")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(angles, norm_db(bart), label='Bartlett (conventional)')
    ax.plot(angles, norm_db(capon), label='MVDR / Capon')
    ax.plot(angles, norm_db(music), label='MUSIC')
    for a in true_angles:
        ax.axvline(a, color='k', ls='--', alpha=0.4)
    ax.set_title('Direction-of-Arrival spectra (16-element line array)',
                 fontweight='bold')
    ax.set_xlabel('Angle from broadside (deg)')
    ax.set_ylabel('Normalised power (dB)')
    ax.set_ylim(-50, 2); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'example_29_array_processing.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("\n✓ Example 29 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
