"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 30: Time-Frequency, Wavenumber & Slowness Transforms Tour
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    uacpy's transform tools side by side, each on the signal it suits:
      • f-k transform   — two plane waves + acoustic-cone line (space-time field)
      • tau-p slant stack — two pulsed events focus to (p, tau) + slowness line
      • Morlet CWT       — multi-component transient (linear time, log frequency)
      • Wigner-Ville     — same transient (sharper, with a cross-term)
      • Cepstrum         — echo-delay (quefrency) recovery
      • Hyperbolic Radon — a curved (NMO) event focuses to (v, tau)

FEATURES DEMONSTRATED:
    ✓ FK (+ draw_sound_cone) · TauP (+ draw_slowness_line)
    ✓ cwt (Morlet) · wigner_ville · cepstrum · radon_transform (hyperbolic)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.acoustic_signal import (  # noqa: E402
    fk_transform,
    taup_transform,
    cepstrum,
    cwt,
    radon_transform,
    wigner_ville,
)
from uacpy.visualization import draw_sound_cone, draw_slowness_line  # noqa: E402


def main():
    print("═" * 80)
    print("EXAMPLE 30: Time-Frequency, Wavenumber & Slowness Transforms Tour")
    print("═" * 80)
    fs = 2000.0

    # --- (A) space-time wavefield for f-k: two plane waves (1500, 2500 m/s) ---
    nt, nx, dx = 256, 48, 1.5
    tt = np.arange(nt) / fs
    xx = np.arange(nx) * dx
    f0 = 200.0
    field = (np.cos(2 * np.pi * (f0 * tt[:, None] - (f0 / 1500.0) * xx[None, :]))
             + 0.8 * np.cos(2 * np.pi * (f0 * tt[:, None] + (f0 / 2500.0) * xx[None, :])))
    field = field * np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    fkf, fkk, fkp, _ = fk_transform(field, fs, dx)

    # --- (B) pulsed linear gather for tau-p ---
    gfs, gnt, gnx, gdx = 1000.0, 512, 48, 10.0
    gt = np.arange(gnt) / gfs
    gx = np.arange(gnx) * gdx

    def ricker(time, tc, F=40.0):
        u = 2 * np.pi * F * (time - tc)
        return (1 - 0.5 * u ** 2) * np.exp(-0.25 * u ** 2)

    gather = np.zeros((gnt, gnx))
    for p0, tau0 in [(1 / 1800.0, 0.04), (-1 / 2500.0, 0.13)]:
        for ix in range(gnx):
            gather[:, ix] += ricker(gt, tau0 + p0 * gx[ix])
    pax, tauax, U = taup_transform(gather, gfs, gdx, p_max=1 / 1200.0,
                                   n_slowness=301)

    # --- (C,D) multi-component transient for CWT + Wigner-Ville ---
    n = 1024
    t = np.arange(n) / fs
    sig = (np.sin(2 * np.pi * 150 * t) * np.exp(-0.5 * ((t - 0.20) / 0.03) ** 2)
           + np.sin(2 * np.pi * 500 * t) * np.exp(-0.5 * ((t - 0.40) / 0.03) ** 2))
    freqs, W = cwt(sig, fs, wavelet='morlet', n_freqs=140)
    fvw, tvw, Wv = wigner_ville(sig, fs)

    # --- (E) broadband pulse + echo for the cepstrum ---
    t0, width, fc = 0.06, 0.004, 250.0
    pulse = np.exp(-((t - t0) / width) ** 2) * np.cos(2 * np.pi * fc * (t - t0))
    echo_delay = 0.040
    d = int(echo_delay * fs)
    xe = pulse.copy()
    xe[d:] += 0.7 * pulse[:-d]
    cep = cepstrum(xe)
    quef = np.arange(cep.size) / fs
    # Search 8-100 ms of quefrency. Below ~8 ms the cepstrum is dominated by
    # the pulse's own spectral envelope (its 4 ms Gaussian width), which would
    # outrank the echo rahmonic the peak search is after.
    lo, hi = int(0.008 * fs), int(0.1 * fs)

    # --- (F) hyperbolic (NMO) gather for the Radon transform ---
    v0, tau0h = 1500.0, 0.07
    ghyp = np.zeros((gnt, gnx))
    for ix in range(gnx):
        ghyp[:, ix] += ricker(gt, np.sqrt(tau0h ** 2 + (gx[ix] / v0) ** 2))
    vels = np.linspace(1200.0, 2000.0, 121)
    _, tauh, Rh = radon_transform(ghyp, gfs, gdx, vels, kind='hyperbolic')

    # U and Rh are (slowness/velocity, tau); a flat argmax floor-divided by the
    # column count recovers the first-axis index, i.e. the winning p or v.
    print(f"\n  tau-p strongest event ≈ {1/abs(pax[np.argmax(np.abs(U)) // U.shape[1]]):.0f} m/s")
    print(f"  cepstrum peak: {quef[lo + np.argmax(cep[lo:hi])]*1e3:.1f} ms "
          f"(echo {echo_delay*1e3:.0f} ms)")
    print(f"  hyperbolic Radon focus: {vels[np.argmax(np.abs(Rh)) // Rh.shape[1]]:.0f} m/s "
          f"(true {v0:.0f})")

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

    # (A) f-k + acoustic cone
    ax = axes[0, 0]
    im = ax.imshow(10 * np.log10(fkp / fkp.max() + 1e-6), aspect='auto',
                   origin='lower', extent=[fkk[0], fkk[-1], fkf[0], fkf[-1]],
                   vmin=-40, vmax=0, cmap='jet')
    draw_sound_cone(ax, fkf[-1], fkk[-1], 1500)
    ax.set_title('f-k transform + 1500 m/s cone', fontweight='bold')
    ax.set_xlabel('Wavenumber k (rad/m)'); ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 400)
    fig.colorbar(im, ax=ax, label='Relative power [dB]')

    # (B) tau-p slant stack + slowness line
    ax = axes[0, 1]
    im = ax.imshow(np.abs(U).T, aspect='auto', origin='upper', cmap='jet',
                   extent=[pax[0] * 1e3, pax[-1] * 1e3, tauax[-1], tauax[0]],
                   vmin=0, vmax=np.abs(U).max())
    draw_slowness_line(ax, tauax[-1], 1500)
    ax.set_title('tau-p slant stack + 1500 m/s', fontweight='bold')
    ax.set_xlabel('Slowness p [s/km]'); ax.set_ylabel('Intercept tau [s]')
    fig.colorbar(im, ax=ax, label='Stack amplitude')

    # (C) Morlet CWT
    ax = axes[1, 0]
    pcm = ax.pcolormesh(t, freqs, np.abs(W), shading='auto', cmap='jet')
    ax.set_yscale('log')
    ax.set_title('Morlet CWT scalogram', fontweight='bold')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')
    fig.colorbar(pcm, ax=ax, label='Wavelet amplitude |W|')

    # (D) Wigner-Ville
    ax = axes[1, 1]
    im = ax.imshow(Wv, aspect='auto', origin='lower',
                   extent=[tvw[0], tvw[-1], fvw[0], fvw[-1]], cmap='jet',
                   vmin=0.0, vmax=Wv.max())
    ax.set_title('Wigner-Ville (cross-term near 325 Hz)', fontweight='bold')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 700)
    fig.colorbar(im, ax=ax, label='Energy density')

    # (E) cepstrum
    ax = axes[2, 0]
    ax.plot(quef[lo:hi] * 1e3, cep[lo:hi], color='#1f77b4', lw=1.2)
    ax.axvline(echo_delay * 1e3, color='crimson', ls='--',
               label=f'echo {echo_delay*1e3:.0f} ms')
    ax.axvline(2 * echo_delay * 1e3, color='crimson', ls=':', alpha=0.6,
               label=f'2× ({2*echo_delay*1e3:.0f} ms)')
    ax.set_title('Cepstrum — echo-delay recovery', fontweight='bold')
    ax.set_xlabel('Quefrency [ms]'); ax.set_ylabel('Amplitude')
    ax.set_xlim(lo / fs * 1e3, hi / fs * 1e3)
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    # (F) hyperbolic Radon
    ax = axes[2, 1]
    im = ax.imshow(np.abs(Rh).T, aspect='auto', origin='upper', cmap='jet',
                   extent=[vels[0], vels[-1], tauh[-1], tauh[0]],
                   vmin=0, vmax=np.abs(Rh).max())
    ax.axvline(v0, color='w', ls='--', lw=1.1)
    ax.set_title('Hyperbolic Radon (velocity)', fontweight='bold')
    ax.set_xlabel('Velocity [m/s]'); ax.set_ylabel('Intercept tau [s]')
    fig.colorbar(im, ax=ax, label='Stack amplitude')

    fig.suptitle('Time-Frequency, Wavenumber & Slowness Transforms',
                 fontsize=15, fontweight='bold')
    out = OUTPUT_DIR / 'example_30_time_frequency.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("\n✓ Example 30 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
